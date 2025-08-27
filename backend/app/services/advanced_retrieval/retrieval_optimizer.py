"""
Retrieval Optimizer for query routing and strategy selection.
Analyzes queries to determine optimal retrieval configuration.
"""

import logging
from typing import Dict, Any, Optional, List
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    length: int
    complexity: float
    query_type: str  # factual, analytical, navigational, etc.
    has_entities: bool
    technical_level: float
    language_pattern: str
    recommended_strategy: Dict[str, Any]


class RetrievalOptimizer:
    """
    Optimizes retrieval strategy based on query analysis.
    Routes queries to appropriate retrieval methods.
    """
    
    def __init__(self):
        """Initialize the retrieval optimizer."""
        # Define patterns for query classification
        self.question_patterns = [
            r'^(what|when|where|who|why|how|which)',
            r'\?$',
            r'^(is|are|was|were|can|could|should|would)',
            r'^(explain|describe|define|list)',
        ]
        
        self.technical_terms = {
            'algorithm', 'method', 'process', 'system', 'model',
            'function', 'implementation', 'architecture', 'framework',
            'protocol', 'interface', 'database', 'network', 'security'
        }
        
        self.entity_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',  # Proper nouns
            r'\b\d{4}\b',  # Years
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        logger.info("RetrievalOptimizer initialized")
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query characteristics.
        
        Args:
            query: User query to analyze
            
        Returns:
            QueryAnalysis object with detailed analysis
        """
        # Basic metrics
        words = query.split()
        length = len(words)
        
        # Complexity score (0-1)
        complexity = self._calculate_complexity(query, words)
        
        # Query type classification
        query_type = self._classify_query_type(query)
        
        # Check for entities
        has_entities = self._has_entities(query)
        
        # Technical level (0-1)
        technical_level = self._calculate_technical_level(query, words)
        
        # Language pattern
        language_pattern = self._detect_language_pattern(query)
        
        # Recommend strategy
        recommended_strategy = self._recommend_strategy(
            length, complexity, query_type, has_entities, technical_level
        )
        
        return QueryAnalysis(
            length=length,
            complexity=complexity,
            query_type=query_type,
            has_entities=has_entities,
            technical_level=technical_level,
            language_pattern=language_pattern,
            recommended_strategy=recommended_strategy
        )
    
    def _calculate_complexity(self, query: str, words: List[str]) -> float:
        """Calculate query complexity score."""
        complexity = 0.0
        
        # Length factor
        if len(words) > 10:
            complexity += 0.3
        elif len(words) > 5:
            complexity += 0.2
        
        # Sentence structure
        if ',' in query or ';' in query:
            complexity += 0.2
        
        # Multiple clauses
        if any(word in words for word in ['and', 'or', 'but', 'because', 'although']):
            complexity += 0.2
        
        # Technical terms
        tech_count = sum(1 for word in words if word.lower() in self.technical_terms)
        complexity += min(tech_count * 0.1, 0.3)
        
        return min(complexity, 1.0)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Check question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, query_lower):
                if query_lower.startswith(('what', 'explain', 'describe')):
                    return 'definitional'
                elif query_lower.startswith(('how', 'why')):
                    return 'explanatory'
                elif query_lower.startswith(('when', 'where', 'who')):
                    return 'factual'
                else:
                    return 'question'
        
        # Check for specific patterns
        if 'compare' in query_lower or 'difference' in query_lower:
            return 'comparative'
        elif 'list' in query_lower or 'examples' in query_lower:
            return 'list'
        elif re.search(r'\b(find|search|looking for)\b', query_lower):
            return 'navigational'
        else:
            return 'general'
    
    def _has_entities(self, query: str) -> bool:
        """Check if query contains named entities."""
        for pattern in self.entity_patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def _calculate_technical_level(self, query: str, words: List[str]) -> float:
        """Calculate technical level of the query."""
        tech_count = sum(1 for word in words if word.lower() in self.technical_terms)
        tech_ratio = tech_count / len(words) if words else 0
        
        # Check for code-like patterns
        if re.search(r'[(){}\[\]]', query) or re.search(r'\w+\(\)', query):
            tech_ratio += 0.3
        
        # Check for technical abbreviations
        if re.search(r'\b[A-Z]{2,}\b', query):
            tech_ratio += 0.2
        
        return min(tech_ratio, 1.0)
    
    def _detect_language_pattern(self, query: str) -> str:
        """Detect the language pattern of the query."""
        if re.search(r'[(){}\[\]]', query) or re.search(r'\w+\(\)', query):
            return 'code'
        elif re.search(r'\?$', query):
            return 'question'
        elif len(query.split()) <= 3:
            return 'keywords'
        else:
            return 'natural'
    
    def _recommend_strategy(
        self,
        length: int,
        complexity: float,
        query_type: str,
        has_entities: bool,
        technical_level: float
    ) -> Dict[str, Any]:
        """
        Recommend retrieval strategy based on analysis.
        
        Returns:
            Dictionary with retrieval configuration
        """
        strategy = {
            'use_bm25': True,
            'use_dense': True,
            'use_hyde': False,
            'use_expansion': False,
            'expansion_methods': [],
            'fusion_strategy': 'rrf',
            'weights': {
                'bm25': 0.33,
                'dense': 0.34,
                'hyde': 0.33
            },
            'reasoning': []
        }
        
        # Short queries benefit from expansion
        if length <= 3:
            strategy['use_expansion'] = True
            strategy['expansion_methods'] = ['bert', 't5']
            strategy['use_hyde'] = True
            strategy['reasoning'].append("Short query - using expansion and HyDE")
        
        # Complex queries need careful handling
        if complexity > 0.7:
            strategy['use_bm25'] = True
            strategy['weights']['bm25'] = 0.4
            strategy['weights']['dense'] = 0.6
            strategy['weights']['hyde'] = 0.0
            strategy['use_hyde'] = False
            strategy['reasoning'].append("Complex query - focusing on exact matching")
        
        # Questions benefit from HyDE
        if query_type in ['question', 'explanatory', 'definitional']:
            strategy['use_hyde'] = True
            strategy['weights']['hyde'] = 0.4
            strategy['weights']['bm25'] = 0.2
            strategy['weights']['dense'] = 0.4
            strategy['reasoning'].append(f"{query_type} query - using HyDE for better understanding")
        
        # Technical queries need precise matching
        if technical_level > 0.5:
            strategy['use_bm25'] = True
            strategy['weights']['bm25'] = 0.5
            strategy['weights']['dense'] = 0.3
            strategy['weights']['hyde'] = 0.2
            strategy['fusion_strategy'] = 'weighted'
            strategy['reasoning'].append("Technical query - emphasizing BM25")
        
        # Entities benefit from exact matching
        if has_entities:
            strategy['use_bm25'] = True
            strategy['weights']['bm25'] = 0.45
            strategy['reasoning'].append("Contains entities - boosting exact match")
        
        # Normalize weights
        total_weight = sum(strategy['weights'].values())
        if total_weight > 0:
            for key in strategy['weights']:
                strategy['weights'][key] /= total_weight
        
        return strategy
    
    def optimize_for_dataset(
        self,
        query: str,
        dataset_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize retrieval strategy based on dataset characteristics.
        
        Args:
            query: User query
            dataset_characteristics: Info about the dataset
            
        Returns:
            Optimized strategy
        """
        # Start with query-based strategy
        analysis = self.analyze_query(query)
        strategy = analysis.recommended_strategy
        
        # Adjust based on dataset
        doc_count = dataset_characteristics.get('doc_count', 0)
        avg_doc_length = dataset_characteristics.get('avg_doc_length', 0)
        domain = dataset_characteristics.get('domain', 'general')
        
        # Large datasets benefit from filtering
        if doc_count > 10000:
            strategy['use_bm25'] = True  # Fast filtering
            strategy['reasoning'].append(f"Large dataset ({doc_count} docs) - using BM25 for efficiency")
        
        # Short documents don't need HyDE
        if avg_doc_length < 100:
            strategy['use_hyde'] = False
            strategy['reasoning'].append("Short documents - skipping HyDE")
        
        # Domain-specific adjustments
        if domain == 'medical':
            strategy['use_hyde'] = True
            strategy['weights']['hyde'] = 0.4
            strategy['reasoning'].append("Medical domain - using HyDE for terminology")
        elif domain == 'legal':
            strategy['use_bm25'] = True
            strategy['weights']['bm25'] = 0.5
            strategy['reasoning'].append("Legal domain - emphasizing exact match")
        elif domain == 'technical':
            strategy['fusion_strategy'] = 'weighted'
            strategy['reasoning'].append("Technical domain - using weighted fusion")
        
        return strategy
    
    def explain_strategy(self, strategy: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the strategy.
        
        Args:
            strategy: Retrieval strategy configuration
            
        Returns:
            Explanation string
        """
        explanation = "Retrieval Strategy:\n"
        
        # Methods being used
        methods = []
        if strategy.get('use_bm25'):
            methods.append(f"BM25 (weight: {strategy['weights']['bm25']:.2f})")
        if strategy.get('use_dense'):
            methods.append(f"Dense (weight: {strategy['weights']['dense']:.2f})")
        if strategy.get('use_hyde'):
            methods.append(f"HyDE (weight: {strategy['weights']['hyde']:.2f})")
        
        explanation += f"• Methods: {', '.join(methods)}\n"
        
        # Query expansion
        if strategy.get('use_expansion'):
            expansion_methods = strategy.get('expansion_methods', [])
            explanation += f"• Query Expansion: {', '.join(expansion_methods)}\n"
        
        # Fusion strategy
        explanation += f"• Fusion: {strategy.get('fusion_strategy', 'rrf')}\n"
        
        # Reasoning
        if strategy.get('reasoning'):
            explanation += "• Reasoning:\n"
            for reason in strategy['reasoning']:
                explanation += f"  - {reason}\n"
        
        return explanation