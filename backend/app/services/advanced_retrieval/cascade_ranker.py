"""
Cascade Reranking Pipeline for multi-stage document ranking.
Implements progressive filtering through multiple ranking stages.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

from .cross_encoder_ranker import CrossEncoderRanker
from .diversity_ranker import DiversityRanker

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for cascade reranking."""
    stages: List[str] = field(default_factory=lambda: ["initial", "cross_encoder", "diversity"])
    stage_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    early_stopping: bool = True
    confidence_threshold: float = 0.9
    progressive_filtering: bool = True
    async_execution: bool = False


@dataclass
class StageResult:
    """Result from a single cascade stage."""
    stage_name: str
    candidates: List[Any]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CascadeRanker:
    """
    Multi-stage cascade reranking pipeline.
    
    Progressively refines search results through multiple ranking stages,
    with each stage potentially reducing the candidate set.
    """
    
    def __init__(
        self,
        config: Optional[CascadeConfig] = None,
        cross_encoder: Optional[CrossEncoderRanker] = None,
        diversity_ranker: Optional[DiversityRanker] = None
    ):
        """
        Initialize cascade ranker.
        
        Args:
            config: Cascade configuration
            cross_encoder: Cross-encoder for reranking stage
            diversity_ranker: Diversity ranker for final stage
        """
        self.config = config or CascadeConfig()
        self.stages = self.config.stages
        self.stage_configs = self.config.stage_configs
        self.early_stopping_enabled = self.config.early_stopping
        
        # Initialize rankers
        self.cross_encoder = cross_encoder
        self.diversity_ranker = diversity_ranker
        
        # Stage execution history
        self.execution_history = []
        
        logger.info(f"CascadeRanker initialized with stages: {self.stages}")
    
    def _initialize_rankers(self):
        """Lazy initialization of rankers."""
        if "cross_encoder" in self.stages and self.cross_encoder is None:
            self.cross_encoder = CrossEncoderRanker()
            
        if "diversity" in self.stages and self.diversity_ranker is None:
            self.diversity_ranker = DiversityRanker()
    
    def execute_stage(
        self,
        stage_name: str,
        query: str,
        candidates: List[Any],
        config: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute a single ranking stage.
        
        Args:
            stage_name: Name of the stage to execute
            query: Search query
            candidates: Current candidate list
            config: Stage-specific configuration
            
        Returns:
            Reranked candidates from this stage
        """
        start_time = time.time()
        
        try:
            if stage_name == "initial":
                # Initial ranking (passthrough or basic filtering)
                results = self._initial_stage(candidates, config)
                
            elif stage_name == "cross_encoder":
                # Cross-encoder reranking
                results = self._cross_encoder_stage(query, candidates, config)
                
            elif stage_name == "diversity":
                # Diversity optimization
                results = self._diversity_stage(query, candidates, config)
                
            elif stage_name == "custom":
                # Custom stage execution
                custom_func = config.get("function")
                if custom_func:
                    results = custom_func(query, candidates, config)
                else:
                    results = candidates
            else:
                logger.warning(f"Unknown stage: {stage_name}")
                results = candidates
            
            # Record execution
            execution_time = time.time() - start_time
            self.execution_history.append(StageResult(
                stage_name=stage_name,
                candidates=results[:5],  # Store only top 5 for history
                execution_time=execution_time,
                metadata={"config": config, "input_size": len(candidates)}
            ))
            
            # Add stage scores to candidates
            for candidate in results:
                if not hasattr(candidate, 'stage_scores'):
                    if isinstance(candidate, dict):
                        candidate['stage_scores'] = {}
                    else:
                        candidate.stage_scores = {}
                
                if isinstance(candidate, dict):
                    candidate['stage_scores'][stage_name] = candidate.get('score', 0)
                else:
                    candidate.stage_scores[stage_name] = getattr(candidate, 'score', 0)
            
            return results
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {str(e)}")
            return candidates  # Return unchanged on error
    
    def _initial_stage(self, candidates: List[Any], config: Dict[str, Any]) -> List[Any]:
        """Initial filtering/ranking stage."""
        top_k = config.get("top_k", len(candidates))
        
        # Sort by initial scores if available
        if candidates and hasattr(candidates[0], 'score'):
            sorted_candidates = sorted(
                candidates,
                key=lambda x: getattr(x, 'score', 0),
                reverse=True
            )
        elif candidates and isinstance(candidates[0], dict) and 'score' in candidates[0]:
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
        else:
            sorted_candidates = candidates
        
        return sorted_candidates[:top_k]
    
    def _cross_encoder_stage(
        self,
        query: str,
        candidates: List[Any],
        config: Dict[str, Any]
    ) -> List[Any]:
        """Cross-encoder reranking stage."""
        if not self.cross_encoder:
            self._initialize_rankers()
        
        top_k = config.get("top_k", len(candidates))
        combine_scores = config.get("combine_scores", True)
        alpha = config.get("alpha", 0.7)
        
        return self.cross_encoder.rerank(
            query,
            candidates,
            top_k=top_k,
            combine_scores=combine_scores,
            alpha=alpha
        )
    
    def _diversity_stage(
        self,
        query: str,
        candidates: List[Any],
        config: Dict[str, Any]
    ) -> List[Any]:
        """Diversity optimization stage."""
        if not self.diversity_ranker:
            self._initialize_rankers()
        
        top_k = config.get("top_k", len(candidates))
        lambda_param = config.get("lambda_param", 0.5)
        method = config.get("method", "mmr")
        
        if method == "mmr":
            return self.diversity_ranker.rerank_mmr(
                query,
                candidates,
                lambda_param=lambda_param,
                top_k=top_k
            )
        elif method == "clustering":
            n_clusters = config.get("n_clusters", 5)
            return self.diversity_ranker.rerank_clustering(
                candidates,
                n_clusters=n_clusters,
                top_k=top_k
            )
        else:
            return candidates[:top_k]
    
    def should_stop_early(
        self,
        stage_name: str,
        candidates: List[Any],
        query: str
    ) -> bool:
        """
        Determine if cascade should stop early.
        
        Args:
            stage_name: Current stage
            candidates: Current candidates
            query: Search query
            
        Returns:
            True if should stop early
        """
        if not self.early_stopping_enabled:
            return False
        
        # Check confidence scores
        if candidates:
            top_score = candidates[0].get('score', 0) if isinstance(candidates[0], dict) else getattr(candidates[0], 'score', 0)
            if top_score > self.config.confidence_threshold:
                logger.info(f"Early stopping at {stage_name}: confidence {top_score:.3f}")
                return True
        
        # Check result quality
        if len(candidates) <= 3:
            logger.info(f"Early stopping at {stage_name}: only {len(candidates)} candidates")
            return True
        
        return False
    
    def should_skip_stage(
        self,
        stage_name: str,
        query: str,
        candidates: List[Any]
    ) -> bool:
        """
        Determine if a stage should be skipped.
        
        Args:
            stage_name: Stage to evaluate
            query: Search query
            candidates: Current candidates
            
        Returns:
            True if stage should be skipped
        """
        # Skip cross-encoder for very short queries
        if stage_name == "cross_encoder" and len(query.split()) <= 2:
            logger.info(f"Skipping {stage_name}: query too short")
            return True
        
        # Skip diversity if few candidates
        if stage_name == "diversity" and len(candidates) <= 5:
            logger.info(f"Skipping {stage_name}: too few candidates")
            return True
        
        return False
    
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        stages: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        early_stopping: Optional[bool] = None
    ) -> List[Any]:
        """
        Execute cascade reranking pipeline.
        
        Args:
            query: Search query
            candidates: Initial candidates
            stages: Stages to execute (default: all)
            top_k: Final number of results
            early_stopping: Override early stopping setting
            
        Returns:
            Final reranked candidates
        """
        stages = stages or self.stages
        early_stopping = early_stopping if early_stopping is not None else self.early_stopping_enabled
        
        # Clear execution history
        self.execution_history.clear()
        
        # Initialize rankers if needed
        self._initialize_rankers()
        
        current_candidates = candidates.copy()
        
        for stage_name in stages:
            # Check if should skip
            if self.should_skip_stage(stage_name, query, current_candidates):
                continue
            
            # Get stage config
            stage_config = self.stage_configs.get(stage_name, {})
            
            # Progressive filtering
            if self.config.progressive_filtering and "top_k" not in stage_config:
                # Progressively reduce candidates
                remaining_stages = len(stages) - stages.index(stage_name)
                stage_config["top_k"] = max(
                    top_k or 10,
                    len(current_candidates) // remaining_stages
                )
            
            # Execute stage
            logger.info(f"Executing stage: {stage_name} with {len(current_candidates)} candidates")
            current_candidates = self.execute_stage(
                stage_name,
                query,
                current_candidates,
                stage_config
            )
            
            # Check early stopping
            if early_stopping and self.should_stop_early(stage_name, current_candidates, query):
                break
        
        # Apply final top-k
        if top_k:
            current_candidates = current_candidates[:top_k]
        
        logger.info(f"Cascade complete: {len(stages)} stages -> {len(current_candidates)} results")
        
        return current_candidates
    
    async def rerank_async(
        self,
        query: str,
        candidates: List[Any],
        stages: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Execute cascade reranking asynchronously.
        
        Args:
            query: Search query
            candidates: Initial candidates
            stages: Stages to execute
            top_k: Final number of results
            
        Returns:
            Final reranked candidates
        """
        # For now, wrap synchronous execution
        # Could be enhanced with true async stages
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rerank,
            query,
            candidates,
            stages,
            top_k
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of cascade execution."""
        if not self.execution_history:
            return {}
        
        total_time = sum(r.execution_time for r in self.execution_history)
        
        return {
            "stages_executed": [r.stage_name for r in self.execution_history],
            "total_time": total_time,
            "stage_times": {
                r.stage_name: r.execution_time
                for r in self.execution_history
            },
            "candidate_reduction": {
                r.stage_name: r.metadata.get("input_size", 0)
                for r in self.execution_history
            }
        }
    
    def optimize_cascade(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize cascade configuration based on performance data.
        
        Args:
            performance_data: Historical performance metrics
            
        Returns:
            Optimized configuration
        """
        # Analyze performance data
        stage_impacts = {}
        stage_costs = {}
        
        for data in performance_data:
            for stage in data.get("stages", []):
                name = stage["name"]
                impact = stage.get("quality_improvement", 0)
                cost = stage.get("latency", 0)
                
                if name not in stage_impacts:
                    stage_impacts[name] = []
                    stage_costs[name] = []
                
                stage_impacts[name].append(impact)
                stage_costs[name].append(cost)
        
        # Calculate average impact and cost
        optimized_stages = []
        for stage in self.stages:
            if stage in stage_impacts:
                avg_impact = np.mean(stage_impacts[stage])
                avg_cost = np.mean(stage_costs[stage])
                
                # Include stage if impact/cost ratio is good
                if avg_impact / (avg_cost + 0.001) > 0.5:
                    optimized_stages.append(stage)
            else:
                optimized_stages.append(stage)  # Keep if no data
        
        return {
            "stages": optimized_stages,
            "stage_configs": self.stage_configs,
            "analysis": {
                "impacts": {k: np.mean(v) for k, v in stage_impacts.items()},
                "costs": {k: np.mean(v) for k, v in stage_costs.items()}
            }
        }