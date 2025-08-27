"""
Comprehensive test suite for Cross-Encoder Reranking.
Tests cross-encoder models, cascade reranking, and MMR diversity.
"""

import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
import torch
from dataclasses import dataclass


@dataclass
class TestCandidate:
    """Test document candidate for reranking."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class TestCrossEncoderRanker:
    """Test suite for Cross-Encoder reranking implementation."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidate documents for reranking."""
        return [
            TestCandidate("1", "Machine learning is a subset of artificial intelligence", 0.8, {"source": "doc1"}),
            TestCandidate("2", "Deep learning uses neural networks with multiple layers", 0.75, {"source": "doc2"}),
            TestCandidate("3", "Natural language processing enables computers to understand text", 0.7, {"source": "doc3"}),
            TestCandidate("4", "Computer vision allows machines to interpret visual information", 0.65, {"source": "doc4"}),
            TestCandidate("5", "Reinforcement learning trains agents through rewards", 0.6, {"source": "doc5"}),
            TestCandidate("6", "Transfer learning leverages pre-trained models", 0.55, {"source": "doc6"}),
            TestCandidate("7", "Supervised learning uses labeled training data", 0.5, {"source": "doc7"}),
            TestCandidate("8", "Unsupervised learning finds patterns in unlabeled data", 0.45, {"source": "doc8"}),
            TestCandidate("9", "Semi-supervised learning combines both approaches", 0.4, {"source": "doc9"}),
            TestCandidate("10", "Active learning selects informative samples for labeling", 0.35, {"source": "doc10"})
        ]
    
    @pytest.fixture
    def cross_encoder(self):
        """Mock cross-encoder ranker."""
        from app.services.advanced_retrieval.cross_encoder_ranker import CrossEncoderRanker
        return CrossEncoderRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def test_cross_encoder_initialization(self, cross_encoder):
        """Test cross-encoder initialization with model loading."""
        assert cross_encoder.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cross_encoder.device in ["cuda", "cpu"]
        assert cross_encoder.batch_size == 32
        assert cross_encoder.max_length == 512
    
    def test_cross_encoder_scoring(self, cross_encoder, sample_candidates):
        """Test cross-encoder relevance scoring."""
        query = "What is machine learning?"
        documents = [c.content for c in sample_candidates]
        
        with patch.object(cross_encoder, 'model') as mock_model:
            # Mock model predictions
            mock_model.predict.return_value = np.array([
                0.95, 0.85, 0.75, 0.65, 0.7, 0.8, 0.9, 0.6, 0.55, 0.5
            ])
            
            scores = cross_encoder.score_pairs(query, documents)
            
            assert len(scores) == len(documents)
            assert all(0 <= s <= 1 for s in scores)
            assert scores[0] > scores[-1]  # First should score higher than last
    
    def test_cross_encoder_batch_processing(self, cross_encoder, sample_candidates):
        """Test batch processing for efficiency."""
        query = "machine learning applications"
        documents = [c.content for c in sample_candidates] * 10  # 100 documents
        
        with patch.object(cross_encoder, 'model') as mock_model:
            mock_scores = np.random.random(100)
            mock_model.predict.return_value = mock_scores
            
            start_time = time.time()
            scores = cross_encoder.score_pairs(query, documents, batch_size=16)
            elapsed = time.time() - start_time
            
            assert len(scores) == 100
            assert elapsed < 2.0  # Should process quickly
            # Check batching was used
            assert mock_model.predict.call_count <= 7  # ceil(100/16)
    
    def test_cross_encoder_reranking(self, cross_encoder, sample_candidates):
        """Test complete reranking pipeline."""
        query = "deep learning neural networks"
        
        with patch.object(cross_encoder, 'score_pairs') as mock_score:
            # Simulate cross-encoder improving ranking
            mock_score.return_value = [0.3, 0.95, 0.7, 0.4, 0.5, 0.6, 0.8, 0.45, 0.35, 0.25]
            
            reranked = cross_encoder.rerank(
                query,
                sample_candidates,
                top_k=5
            )
            
            assert len(reranked) == 5
            assert reranked[0].doc_id == "2"  # Deep learning doc should rank first
            assert all(hasattr(r, 'cross_encoder_score') for r in reranked)
    
    def test_cross_encoder_score_calibration(self, cross_encoder):
        """Test score calibration for interpretability."""
        raw_scores = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
        
        calibrated = cross_encoder.calibrate_scores(raw_scores)
        
        assert len(calibrated) == len(raw_scores)
        assert all(0 <= s <= 1 for s in calibrated)
        assert calibrated[0] < calibrated[-1]  # Monotonic
    
    def test_cross_encoder_caching(self, cross_encoder):
        """Test caching of cross-encoder scores."""
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]
        
        with patch.object(cross_encoder, 'model') as mock_model:
            mock_model.predict.return_value = np.array([0.8, 0.7, 0.6])
            
            # First call
            scores1 = cross_encoder.score_pairs(query, documents, use_cache=True)
            call_count1 = mock_model.predict.call_count
            
            # Second call (should use cache)
            scores2 = cross_encoder.score_pairs(query, documents, use_cache=True)
            call_count2 = mock_model.predict.call_count
            
            assert np.array_equal(scores1, scores2)
            assert call_count2 == call_count1  # No additional model calls
    
    def test_cross_encoder_error_handling(self, cross_encoder):
        """Test error handling in cross-encoder."""
        query = "test query"
        
        with patch.object(cross_encoder, 'model') as mock_model:
            mock_model.predict.side_effect = RuntimeError("CUDA out of memory")
            
            # Should fallback gracefully
            scores = cross_encoder.score_pairs(query, ["doc1", "doc2"])
            
            assert scores is not None
            assert len(scores) == 2


class TestCascadeRanker:
    """Test suite for cascade reranking pipeline."""
    
    @pytest.fixture
    def cascade_ranker(self):
        """Mock cascade reranking pipeline."""
        from app.services.advanced_retrieval.cascade_ranker import CascadeRanker
        return CascadeRanker()
    
    def test_cascade_initialization(self, cascade_ranker):
        """Test cascade ranker initialization."""
        assert cascade_ranker.stages == ["initial", "cross_encoder", "diversity"]
        assert cascade_ranker.stage_configs is not None
        assert cascade_ranker.early_stopping_enabled == True
    
    def test_cascade_single_stage(self, cascade_ranker, sample_candidates):
        """Test single stage execution."""
        query = "machine learning"
        
        # Test initial stage only
        results = cascade_ranker.rerank(
            query,
            sample_candidates,
            stages=["initial"]
        )
        
        assert len(results) <= len(sample_candidates)
        assert all(hasattr(r, 'stage_scores') for r in results)
        assert "initial" in results[0].stage_scores
    
    def test_cascade_multi_stage(self, cascade_ranker, sample_candidates):
        """Test multi-stage cascade execution."""
        query = "deep learning applications"
        
        with patch.object(cascade_ranker, 'cross_encoder_stage') as mock_ce, \
             patch.object(cascade_ranker, 'diversity_stage') as mock_div:
            
            # Mock stage outputs
            mock_ce.return_value = sample_candidates[:5]
            mock_div.return_value = sample_candidates[:3]
            
            results = cascade_ranker.rerank(
                query,
                sample_candidates,
                stages=["initial", "cross_encoder", "diversity"],
                top_k=3
            )
            
            assert len(results) == 3
            mock_ce.assert_called_once()
            mock_div.assert_called_once()
    
    def test_cascade_progressive_filtering(self, cascade_ranker, sample_candidates):
        """Test progressive candidate reduction."""
        query = "test query"
        
        # Configure progressive filtering
        cascade_ranker.stage_configs = {
            "initial": {"top_k": 10},
            "cross_encoder": {"top_k": 5},
            "diversity": {"top_k": 3}
        }
        
        with patch.object(cascade_ranker, 'execute_stage') as mock_execute:
            def execute_with_filtering(stage, query, candidates, config):
                return candidates[:config.get("top_k", len(candidates))]
            
            mock_execute.side_effect = execute_with_filtering
            
            results = cascade_ranker.rerank(
                query,
                sample_candidates,
                stages=["initial", "cross_encoder", "diversity"]
            )
            
            # Should progressively reduce candidates
            assert len(results) == 3
    
    def test_cascade_early_stopping(self, cascade_ranker, sample_candidates):
        """Test early stopping based on confidence."""
        query = "very specific query"
        
        with patch.object(cascade_ranker, 'should_stop_early') as mock_stop:
            mock_stop.return_value = True
            
            results = cascade_ranker.rerank(
                query,
                sample_candidates,
                stages=["initial", "cross_encoder", "diversity"],
                early_stopping=True
            )
            
            # Should stop after first stage
            assert len(results) > 0
            assert "cross_encoder" not in results[0].stage_scores
    
    def test_cascade_stage_skipping(self, cascade_ranker, sample_candidates):
        """Test conditional stage skipping."""
        query = "simple keyword search"
        
        with patch.object(cascade_ranker, 'should_skip_stage') as mock_skip:
            mock_skip.side_effect = [False, True, False]  # Skip cross-encoder
            
            results = cascade_ranker.rerank(
                query,
                sample_candidates,
                stages=["initial", "cross_encoder", "diversity"]
            )
            
            assert "cross_encoder" not in results[0].stage_scores
            assert "diversity" in results[0].stage_scores
    
    @pytest.mark.asyncio
    async def test_cascade_async_execution(self, cascade_ranker, sample_candidates):
        """Test asynchronous cascade execution."""
        query = "async test"
        
        results = await cascade_ranker.rerank_async(
            query,
            sample_candidates,
            stages=["initial", "cross_encoder"]
        )
        
        assert len(results) > 0
        assert results[0].doc_id in [c.doc_id for c in sample_candidates]


class TestDiversityRanker:
    """Test suite for diversity optimization (MMR)."""
    
    @pytest.fixture
    def diversity_ranker(self):
        """Mock diversity ranker with MMR."""
        from app.services.advanced_retrieval.diversity_ranker import DiversityRanker
        return DiversityRanker()
    
    @pytest.fixture
    def similar_candidates(self):
        """Create candidates with varying similarity."""
        return [
            TestCandidate("1", "Python is a programming language", 0.9, {}),
            TestCandidate("2", "Python is a coding language", 0.85, {}),  # Very similar to 1
            TestCandidate("3", "Java is a programming language", 0.8, {}),
            TestCandidate("4", "JavaScript is used for web development", 0.75, {}),
            TestCandidate("5", "Python programming is popular", 0.7, {}),  # Similar to 1,2
            TestCandidate("6", "Machine learning uses Python", 0.65, {}),
            TestCandidate("7", "Data science requires statistics", 0.6, {}),
            TestCandidate("8", "Cloud computing is scalable", 0.55, {}),
        ]
    
    def test_mmr_initialization(self, diversity_ranker):
        """Test MMR ranker initialization."""
        assert diversity_ranker.lambda_param == 0.5
        assert diversity_ranker.similarity_threshold == 0.8
        assert diversity_ranker.method == "mmr"
    
    def test_mmr_diversity_scoring(self, diversity_ranker, similar_candidates):
        """Test MMR diversity scoring."""
        query = "programming languages"
        
        with patch.object(diversity_ranker, 'compute_similarity') as mock_sim:
            # Mock similarity matrix
            mock_sim.side_effect = lambda d1, d2: 0.95 if d1.doc_id == "1" and d2.doc_id == "2" else 0.3
            
            results = diversity_ranker.rerank_mmr(
                query,
                similar_candidates,
                lambda_param=0.5,
                top_k=4
            )
            
            assert len(results) == 4
            # Should not have both doc 1 and 2 (too similar)
            doc_ids = [r.doc_id for r in results]
            assert not (("1" in doc_ids) and ("2" in doc_ids))
    
    def test_mmr_lambda_parameter(self, diversity_ranker, similar_candidates):
        """Test effect of lambda parameter on diversity."""
        query = "programming"
        
        # Lambda = 1.0 (only relevance)
        results_relevance = diversity_ranker.rerank_mmr(
            query,
            similar_candidates,
            lambda_param=1.0,
            top_k=3
        )
        
        # Lambda = 0.0 (only diversity)
        results_diversity = diversity_ranker.rerank_mmr(
            query,
            similar_candidates,
            lambda_param=0.0,
            top_k=3
        )
        
        # Results should differ
        relevance_ids = [r.doc_id for r in results_relevance]
        diversity_ids = [r.doc_id for r in results_diversity]
        assert relevance_ids != diversity_ids
    
    def test_clustering_diversity(self, diversity_ranker, similar_candidates):
        """Test clustering-based diversity."""
        with patch.object(diversity_ranker, 'cluster_documents') as mock_cluster:
            # Mock cluster assignments
            mock_cluster.return_value = [0, 0, 1, 1, 0, 2, 2, 3]  # 4 clusters
            
            results = diversity_ranker.rerank_clustering(
                similar_candidates,
                n_clusters=4,
                top_k=4
            )
            
            assert len(results) == 4
            # Should have one from each cluster
            cluster_ids = [mock_cluster.return_value[int(r.doc_id) - 1] for r in results]
            assert len(set(cluster_ids)) == 4
    
    def test_coverage_optimization(self, diversity_ranker, similar_candidates):
        """Test topic coverage optimization."""
        # Add topic metadata
        for i, candidate in enumerate(similar_candidates):
            candidate.metadata['topic'] = ['programming', 'languages', 'python'][i % 3]
        
        results = diversity_ranker.rerank_coverage(
            similar_candidates,
            coverage_field='topic',
            top_k=6
        )
        
        # Should cover all topics
        topics = set()
        for r in results:
            topics.add(r.metadata.get('topic'))
        assert len(topics) >= 2  # Should have multiple topics
    
    def test_diversity_metrics(self, diversity_ranker, similar_candidates):
        """Test diversity metric calculation."""
        selected = similar_candidates[:4]
        
        metrics = diversity_ranker.calculate_diversity_metrics(selected)
        
        assert 'avg_similarity' in metrics
        assert 'min_similarity' in metrics
        assert 'coverage_score' in metrics
        assert 0 <= metrics['avg_similarity'] <= 1
    
    def test_incremental_mmr(self, diversity_ranker, similar_candidates):
        """Test incremental MMR selection."""
        query = "programming"
        selected = []
        remaining = similar_candidates.copy()
        
        for _ in range(3):
            next_doc = diversity_ranker.select_next_mmr(
                query,
                selected,
                remaining,
                lambda_param=0.5
            )
            assert next_doc is not None
            selected.append(next_doc)
            remaining.remove(next_doc)
        
        assert len(selected) == 3
        assert all(s not in remaining for s in selected)


class TestRerankerPerformance:
    """Performance and integration tests for reranking."""
    
    @pytest.fixture
    def integrated_reranker(self):
        """Create integrated reranking system."""
        from app.services.advanced_retrieval.integrated_reranker import IntegratedReranker
        return IntegratedReranker()
    
    def test_reranker_latency(self, integrated_reranker, sample_candidates):
        """Test reranking latency requirements."""
        query = "test query"
        candidates = sample_candidates * 10  # 100 candidates
        
        start_time = time.time()
        results = integrated_reranker.rerank(
            query,
            candidates,
            use_cross_encoder=True,
            use_diversity=True,
            top_k=10
        )
        elapsed = time.time() - start_time
        
        assert len(results) == 10
        assert elapsed < 0.5  # Should complete within 500ms
    
    def test_reranker_quality_improvement(self, integrated_reranker):
        """Test reranking quality improvements."""
        query = "machine learning algorithms"
        
        # Create candidates with known relevance
        candidates = [
            TestCandidate("1", "Unrelated content about cooking", 0.9, {}),  # High initial score but irrelevant
            TestCandidate("2", "Machine learning algorithms explained", 0.4, {}),  # Low score but relevant
            TestCandidate("3", "Deep learning and neural networks", 0.5, {}),
            TestCandidate("4", "Random text about sports", 0.8, {}),
            TestCandidate("5", "Supervised learning algorithms", 0.3, {}),
        ]
        
        with patch.object(integrated_reranker, 'cross_encoder') as mock_ce:
            # Mock cross-encoder fixing the ranking
            mock_ce.score_pairs.return_value = [0.1, 0.95, 0.8, 0.05, 0.85]
            
            results = integrated_reranker.rerank(
                query,
                candidates,
                use_cross_encoder=True,
                top_k=3
            )
            
            # Relevant documents should rank higher
            assert results[0].doc_id == "2"
            assert results[1].doc_id in ["3", "5"]
            assert "1" not in [r.doc_id for r in results]  # Irrelevant filtered out
    
    def test_reranker_memory_usage(self, integrated_reranker):
        """Test memory efficiency of reranking."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch
        query = "test"
        large_candidates = [
            TestCandidate(str(i), f"Document content {i}" * 100, 0.5, {})
            for i in range(1000)
        ]
        
        results = integrated_reranker.rerank(
            query,
            large_candidates,
            batch_size=50,
            top_k=10
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert len(results) == 10
        assert memory_increase < 500  # Should use less than 500MB
    
    @pytest.mark.asyncio
    async def test_concurrent_reranking(self, integrated_reranker, sample_candidates):
        """Test concurrent reranking requests."""
        queries = [f"query {i}" for i in range(10)]
        
        async def rerank_query(query):
            return integrated_reranker.rerank(
                query,
                sample_candidates,
                top_k=5
            )
        
        start_time = time.time()
        results = await asyncio.gather(*[rerank_query(q) for q in queries])
        elapsed = time.time() - start_time
        
        assert len(results) == 10
        assert all(len(r) == 5 for r in results)
        assert elapsed < 2.0  # Should handle 10 concurrent requests quickly


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])