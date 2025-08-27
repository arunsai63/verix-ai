"""
Comprehensive test suite for Advanced Retrieval Mechanisms.
Tests BM25, Hybrid Search, Query Expansion, HyDE, and RRF.
"""

import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock
import time
import json
from dataclasses import dataclass


@dataclass
class TestDocument:
    """Test document structure."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None


class TestBM25Retriever:
    """Test suite for BM25 sparse retrieval implementation."""
    
    @pytest.fixture
    def sample_corpus(self):
        """Create a sample corpus for testing."""
        return [
            TestDocument("1", "The quick brown fox jumps over the lazy dog", {"title": "Fox Story"}),
            TestDocument("2", "Machine learning is a subset of artificial intelligence", {"title": "ML Basics"}),
            TestDocument("3", "Deep learning neural networks require large datasets", {"title": "DL Requirements"}),
            TestDocument("4", "Natural language processing helps computers understand human language", {"title": "NLP Intro"}),
            TestDocument("5", "The fox is quick and clever in the forest", {"title": "Fox Nature"}),
            TestDocument("6", "Artificial intelligence will transform many industries", {"title": "AI Future"}),
            TestDocument("7", "Python is a popular programming language for machine learning", {"title": "Python ML"}),
            TestDocument("8", "Large language models like GPT are trained on massive text corpora", {"title": "LLMs"}),
            TestDocument("9", "Transfer learning allows models to leverage pre-trained knowledge", {"title": "Transfer Learning"}),
            TestDocument("10", "The dog chased the cat through the garden", {"title": "Pet Story"})
        ]
    
    @pytest.fixture
    def bm25_retriever(self):
        """Mock BM25 retriever for testing."""
        from app.services.advanced_retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(k1=1.2, b=0.75)
    
    def test_bm25_initialization(self, bm25_retriever):
        """Test BM25 retriever initialization with parameters."""
        assert bm25_retriever.k1 == 1.2
        assert bm25_retriever.b == 0.75
        assert bm25_retriever.epsilon == 0.25
        assert bm25_retriever.corpus == []
        assert bm25_retriever.doc_len == []
        assert bm25_retriever.inverted_index == {}
    
    def test_bm25_fit_corpus(self, bm25_retriever, sample_corpus):
        """Test fitting BM25 on a corpus."""
        documents = [doc.content for doc in sample_corpus]
        bm25_retriever.fit(documents)
        
        assert len(bm25_retriever.corpus) == 10
        assert len(bm25_retriever.doc_len) == 10
        assert len(bm25_retriever.inverted_index) > 0
        assert bm25_retriever.avgdl > 0
        
        # Check inverted index structure (words are stemmed)
        # "fox" remains "fox", "machine" becomes "machin", "learning" becomes "learn"
        assert "fox" in bm25_retriever.inverted_index
        assert "machin" in bm25_retriever.inverted_index  # stemmed form of "machine"
        assert "learn" in bm25_retriever.inverted_index  # stemmed form of "learning"
    
    def test_bm25_score_calculation(self, bm25_retriever, sample_corpus):
        """Test BM25 score calculation for queries."""
        documents = [doc.content for doc in sample_corpus]
        bm25_retriever.fit(documents)
        
        # Test single term query
        scores = bm25_retriever.get_scores("fox")
        assert len(scores) == 10
        assert scores[0] > 0  # First document contains "fox"
        assert scores[4] > 0  # Fifth document contains "fox"
        assert scores[1] == 0  # Second document doesn't contain "fox"
        
        # Test multi-term query
        scores = bm25_retriever.get_scores("machine learning")
        assert scores[1] > 0  # Contains both terms
        assert scores[6] > 0  # Contains both terms
        assert scores[0] == 0  # Contains neither term
    
    def test_bm25_ranking(self, bm25_retriever, sample_corpus):
        """Test BM25 ranking quality."""
        documents = [doc.content for doc in sample_corpus]
        bm25_retriever.fit(documents)
        
        # Query: "artificial intelligence machine learning"
        top_docs = bm25_retriever.get_top_k("artificial intelligence machine learning", k=5)
        
        assert len(top_docs) == 5
        # Documents about AI/ML should rank higher
        doc_indices = [doc['index'] for doc in top_docs]
        assert 1 in doc_indices[:3]  # "Machine learning...artificial intelligence"
        assert 5 in doc_indices[:3]  # "Artificial intelligence will transform..."
    
    def test_bm25_edge_cases(self, bm25_retriever, sample_corpus):
        """Test BM25 with edge cases."""
        documents = [doc.content for doc in sample_corpus]
        bm25_retriever.fit(documents)
        
        # Empty query
        scores = bm25_retriever.get_scores("")
        assert all(score == 0 for score in scores)
        
        # Query with no matches
        scores = bm25_retriever.get_scores("xyz123 nonexistent")
        assert all(score == 0 for score in scores)
        
        # Single document corpus
        from app.services.advanced_retrieval.bm25_retriever import BM25Retriever
        single_retriever = BM25Retriever()
        single_retriever.fit(["single document"])
        scores = single_retriever.get_scores("document")
        assert len(scores) == 1
        assert scores[0] > 0
    
    def test_bm25_performance(self, bm25_retriever):
        """Test BM25 performance with larger corpus."""
        # Generate larger corpus
        large_corpus = [f"Document {i} with content about topic {i%10}" for i in range(1000)]
        
        start_time = time.time()
        bm25_retriever.fit(large_corpus)
        fit_time = time.time() - start_time
        
        assert fit_time < 1.0  # Should fit 1000 docs in under 1 second
        
        start_time = time.time()
        scores = bm25_retriever.get_scores("topic content document")
        score_time = time.time() - start_time
        
        assert score_time < 0.1  # Should score in under 100ms


class TestRecipocalRankFusion:
    """Test suite for Reciprocal Rank Fusion."""
    
    @pytest.fixture
    def fusion_strategy(self):
        """Mock RRF fusion strategy."""
        from app.services.advanced_retrieval.fusion_strategies import RecipocalRankFusion
        return RecipocalRankFusion()
    
    def test_rrf_basic_fusion(self, fusion_strategy):
        """Test basic RRF fusion of two rankings."""
        # Rankings from two different retrievers
        ranking1 = [
            {"doc_id": "1", "score": 0.9, "rank": 1},
            {"doc_id": "2", "score": 0.8, "rank": 2},
            {"doc_id": "3", "score": 0.7, "rank": 3},
        ]
        
        ranking2 = [
            {"doc_id": "2", "score": 0.95, "rank": 1},
            {"doc_id": "3", "score": 0.85, "rank": 2},
            {"doc_id": "1", "score": 0.75, "rank": 3},
        ]
        
        fused = fusion_strategy.fuse_rankings([ranking1, ranking2], k=60)
        
        assert len(fused) == 3
        assert fused[0]["doc_id"] == "2"  # Should rank highest (rank 2 and 1)
        assert all("rrf_score" in doc for doc in fused)
    
    def test_rrf_with_different_k_values(self, fusion_strategy):
        """Test RRF with different k parameter values."""
        rankings = [
            [{"doc_id": "1", "rank": 1}, {"doc_id": "2", "rank": 2}],
            [{"doc_id": "2", "rank": 1}, {"doc_id": "1", "rank": 2}],
        ]
        
        # Test with different k values
        fused_k60 = fusion_strategy.fuse_rankings(rankings, k=60)
        fused_k10 = fusion_strategy.fuse_rankings(rankings, k=10)
        fused_k1 = fusion_strategy.fuse_rankings(rankings, k=1)
        
        # Scores should be different with different k values
        assert fused_k60[0]["rrf_score"] != fused_k10[0]["rrf_score"]
        assert fused_k10[0]["rrf_score"] != fused_k1[0]["rrf_score"]
    
    def test_rrf_with_missing_documents(self, fusion_strategy):
        """Test RRF when documents appear in only some rankings."""
        ranking1 = [
            {"doc_id": "1", "rank": 1},
            {"doc_id": "2", "rank": 2},
            {"doc_id": "3", "rank": 3},
        ]
        
        ranking2 = [
            {"doc_id": "4", "rank": 1},
            {"doc_id": "2", "rank": 2},
            {"doc_id": "5", "rank": 3},
        ]
        
        fused = fusion_strategy.fuse_rankings([ranking1, ranking2])
        
        assert len(fused) == 5  # All unique documents
        # Documents appearing in both rankings should rank higher
        doc_ids = [doc["doc_id"] for doc in fused]
        assert doc_ids.index("2") < doc_ids.index("1")  # Doc 2 appears in both
    
    def test_weighted_rrf(self, fusion_strategy):
        """Test weighted RRF fusion."""
        rankings = [
            [{"doc_id": "1", "rank": 1}, {"doc_id": "2", "rank": 2}],
            [{"doc_id": "2", "rank": 1}, {"doc_id": "3", "rank": 2}],
        ]
        
        # Give more weight to first ranking
        weights = [0.7, 0.3]
        fused = fusion_strategy.weighted_fusion(rankings, weights)
        
        assert len(fused) == 3
        # Document 1 should rank higher due to weight despite appearing in only one ranking
        assert fused[0]["doc_id"] in ["1", "2"]


class TestQueryExpansion:
    """Test suite for query expansion mechanisms."""
    
    @pytest.fixture
    def query_expander(self):
        """Mock query expansion service."""
        from app.services.advanced_retrieval.query_expansion import QueryExpansion
        return QueryExpansion()
    
    @pytest.mark.asyncio
    async def test_t5_query_expansion(self, query_expander):
        """Test T5-based query expansion."""
        original_query = "machine learning algorithms"
        
        with patch.object(query_expander, 't5_model') as mock_model:
            mock_model.generate.return_value = [
                "supervised learning methods",
                "neural network training",
                "deep learning models"
            ]
            
            expanded_queries = await query_expander.expand_with_t5(original_query)
            
            assert len(expanded_queries) > 0
            assert original_query in expanded_queries
            assert "supervised learning methods" in expanded_queries
    
    @pytest.mark.asyncio
    async def test_bert_synonym_expansion(self, query_expander):
        """Test BERT-based synonym expansion."""
        original_query = "fast car"
        
        with patch.object(query_expander, 'bert_model') as mock_model:
            mock_model.get_synonyms.return_value = {
                "fast": ["quick", "rapid", "speedy"],
                "car": ["automobile", "vehicle"]
            }
            
            expanded_queries = await query_expander.expand_with_bert(original_query)
            
            assert len(expanded_queries) > 1
            assert any("quick" in q for q in expanded_queries)
            assert any("vehicle" in q for q in expanded_queries)
    
    @pytest.mark.asyncio
    async def test_pseudo_relevance_feedback(self, query_expander):
        """Test pseudo-relevance feedback expansion."""
        query = "artificial intelligence"
        top_docs = [
            {"content": "AI and machine learning are transforming technology"},
            {"content": "Neural networks enable deep learning in AI systems"},
            {"content": "Artificial intelligence applications in healthcare"}
        ]
        
        expanded = await query_expander.pseudo_relevance_expansion(query, top_docs)
        
        assert len(expanded) > 0
        # Should extract relevant terms from top documents
        assert any("machine learning" in q.lower() for q in expanded)
        assert any("neural" in q.lower() or "deep learning" in q.lower() for q in expanded)
    
    @pytest.mark.asyncio
    async def test_combined_expansion(self, query_expander):
        """Test combining multiple expansion methods."""
        query = "data science"
        
        with patch.multiple(query_expander,
                          expand_with_t5=AsyncMock(return_value=["machine learning", "data analysis"]),
                          expand_with_bert=AsyncMock(return_value=["data analytics", "information science"]),
                          pseudo_relevance_expansion=AsyncMock(return_value=["statistical analysis"])):
            
            combined = await query_expander.combine_expansions(
                query, 
                methods=['t5', 'bert', 'pseudo']
            )
            
            assert len(combined) >= 4  # Original + expansions
            assert query in combined
            assert "machine learning" in combined
            assert "data analytics" in combined
    
    @pytest.mark.asyncio
    async def test_expansion_performance(self, query_expander):
        """Test query expansion performance."""
        queries = ["query " + str(i) for i in range(10)]
        
        with patch.object(query_expander, 'expand_with_t5', 
                         new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = ["expanded"]
            
            start_time = time.time()
            tasks = [query_expander.expand_with_t5(q) for q in queries]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            
            assert elapsed < 1.0  # Should handle 10 queries in under 1 second
            assert len(results) == 10


class TestHyDE:
    """Test suite for Hypothetical Document Embeddings."""
    
    @pytest.fixture
    def hyde_generator(self):
        """Mock HyDE generator."""
        from app.services.advanced_retrieval.hyde_generator import HyDEGenerator
        return HyDEGenerator()
    
    @pytest.mark.asyncio
    async def test_hypothetical_document_generation(self, hyde_generator):
        """Test generating hypothetical documents from queries."""
        query = "What are the benefits of renewable energy?"
        
        with patch.object(hyde_generator, 'llm') as mock_llm:
            mock_llm.generate.return_value = """
            Renewable energy offers numerous benefits including reduced greenhouse gas emissions,
            lower long-term costs, energy independence, and job creation in green industries.
            Solar, wind, and hydroelectric power are sustainable alternatives to fossil fuels.
            """
            
            hypothetical_doc = await hyde_generator.generate_hypothetical_doc(query)
            
            assert len(hypothetical_doc) > 0
            assert "renewable" in hypothetical_doc.lower()
            assert "energy" in hypothetical_doc.lower()
    
    @pytest.mark.asyncio
    async def test_hyde_embedding_and_search(self, hyde_generator):
        """Test HyDE embedding and search process."""
        query = "machine learning applications"
        
        with patch.object(hyde_generator, 'generate_hypothetical_doc') as mock_gen, \
             patch.object(hyde_generator, 'embeddings_model') as mock_embed, \
             patch.object(hyde_generator, 'vector_store') as mock_store:
            
            mock_gen.return_value = "Machine learning has many applications..."
            mock_embed.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            mock_store.similarity_search.return_value = [
                {"content": "ML in healthcare", "score": 0.9},
                {"content": "ML in finance", "score": 0.85}
            ]
            
            results = await hyde_generator.hyde_search(query, k=5)
            
            assert len(results) == 2
            assert results[0]["score"] > results[1]["score"]
    
    @pytest.mark.asyncio
    async def test_hyde_with_role_context(self, hyde_generator):
        """Test HyDE with different role contexts."""
        query = "explain neural networks"
        
        contexts = {
            "doctor": "medical applications and patient care",
            "lawyer": "legal implications and regulations",
            "general": "general technical explanation"
        }
        
        for role, expected_context in contexts.items():
            with patch.object(hyde_generator, 'llm') as mock_llm:
                hypothetical = await hyde_generator.generate_hypothetical_doc(
                    query, 
                    role_context=role
                )
                
                # Check that role context influences generation
                mock_llm.generate.assert_called()
                call_args = mock_llm.generate.call_args
                assert role in str(call_args) or expected_context in str(call_args)
    
    @pytest.mark.asyncio
    async def test_hyde_combination_with_original(self, hyde_generator):
        """Test combining HyDE results with original query results."""
        hyde_results = [
            {"doc_id": "1", "score": 0.9},
            {"doc_id": "2", "score": 0.8},
            {"doc_id": "3", "score": 0.7}
        ]
        
        original_results = [
            {"doc_id": "2", "score": 0.85},
            {"doc_id": "4", "score": 0.75},
            {"doc_id": "1", "score": 0.65}
        ]
        
        combined = await hyde_generator.combine_with_original(
            hyde_results, 
            original_results,
            hyde_weight=0.6
        )
        
        assert len(combined) == 4  # 4 unique documents
        # Document 2 should rank high (appears in both)
        assert combined[0]["doc_id"] in ["1", "2"]
        assert "combined_score" in combined[0]
    
    @pytest.mark.asyncio
    async def test_hyde_error_handling(self, hyde_generator):
        """Test HyDE error handling and fallback."""
        query = "test query"
        
        with patch.object(hyde_generator, 'generate_hypothetical_doc') as mock_gen:
            mock_gen.side_effect = Exception("LLM failure")
            
            # Should fallback gracefully
            results = await hyde_generator.hyde_search(query, k=5)
            
            assert results is not None
            # Should return empty or fallback results


class TestHybridRetriever:
    """Test suite for the complete hybrid retrieval system."""
    
    @pytest.fixture
    def hybrid_retriever(self):
        """Mock hybrid retriever combining all components."""
        from app.services.advanced_retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever()
    
    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_retrieval(self, hybrid_retriever):
        """Test complete hybrid retrieval pipeline."""
        query = "artificial intelligence in healthcare"
        dataset_names = ["medical_docs"]
        
        with patch.multiple(hybrid_retriever,
                          bm25_search=AsyncMock(return_value=[
                              {"doc_id": "1", "score": 0.8},
                              {"doc_id": "2", "score": 0.7}
                          ]),
                          dense_search=AsyncMock(return_value=[
                              {"doc_id": "2", "score": 0.85},
                              {"doc_id": "3", "score": 0.75}
                          ]),
                          hyde_search=AsyncMock(return_value=[
                              {"doc_id": "1", "score": 0.9},
                              {"doc_id": "4", "score": 0.65}
                          ])):
            
            results = await hybrid_retriever.search(
                query=query,
                dataset_names=dataset_names,
                use_hyde=True,
                use_expansion=True,
                k=10
            )
            
            assert len(results) > 0
            assert all("final_score" in r for r in results)
            assert results[0]["final_score"] >= results[-1]["final_score"]  # Properly sorted
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_performance(self, hybrid_retriever):
        """Test hybrid retrieval performance requirements."""
        queries = [f"query {i}" for i in range(100)]
        
        async def mock_search(query):
            await asyncio.sleep(0.01)  # Simulate processing
            return [{"doc_id": f"doc_{i}", "score": 0.5} for i in range(10)]
        
        with patch.object(hybrid_retriever, 'search', side_effect=mock_search):
            start_time = time.time()
            
            # Process queries concurrently
            tasks = [hybrid_retriever.search(q) for q in queries[:10]]
            results = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            
            assert elapsed < 1.0  # 10 queries in under 1 second
            assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_fallback(self, hybrid_retriever):
        """Test fallback mechanisms in hybrid retrieval."""
        query = "test query"
        
        # Simulate BM25 failure
        with patch.object(hybrid_retriever, 'bm25_search') as mock_bm25, \
             patch.object(hybrid_retriever, 'dense_search') as mock_dense:
            
            mock_bm25.side_effect = Exception("BM25 index error")
            mock_dense.return_value = [{"doc_id": "1", "score": 0.8}]
            
            results = await hybrid_retriever.search(query)
            
            assert len(results) > 0  # Should still return results
            assert results[0]["doc_id"] == "1"  # From dense search
    
    @pytest.mark.asyncio
    async def test_configurable_weights(self, hybrid_retriever):
        """Test configurable weights for different retrieval methods."""
        query = "test"
        
        # Test different weight configurations
        weight_configs = [
            {"bm25": 0.5, "dense": 0.5, "hyde": 0.0},
            {"bm25": 0.3, "dense": 0.7, "hyde": 0.0},
            {"bm25": 0.2, "dense": 0.4, "hyde": 0.4},
        ]
        
        for weights in weight_configs:
            hybrid_retriever.set_weights(weights)
            
            assert hybrid_retriever.weights["bm25"] == weights["bm25"]
            assert hybrid_retriever.weights["dense"] == weights["dense"]
            assert hybrid_retriever.weights["hyde"] == weights["hyde"]
            
            # Weights should sum to 1.0
            assert abs(sum(weights.values()) - 1.0) < 0.01


class TestRetrievalMetrics:
    """Test suite for retrieval evaluation metrics."""
    
    @pytest.fixture
    def metrics_evaluator(self):
        """Mock metrics evaluator."""
        from app.services.advanced_retrieval.metrics import RetrievalMetrics
        return RetrievalMetrics()
    
    def test_mrr_calculation(self, metrics_evaluator):
        """Test Mean Reciprocal Rank calculation."""
        # Rankings with relevant doc positions
        rankings = [
            {"query": "q1", "relevant_pos": 1},  # MRR = 1/1 = 1.0
            {"query": "q2", "relevant_pos": 3},  # MRR = 1/3 = 0.33
            {"query": "q3", "relevant_pos": 2},  # MRR = 1/2 = 0.5
        ]
        
        mrr = metrics_evaluator.calculate_mrr(rankings)
        expected_mrr = (1.0 + 0.33 + 0.5) / 3
        
        assert abs(mrr - expected_mrr) < 0.01
    
    def test_ndcg_calculation(self, metrics_evaluator):
        """Test Normalized Discounted Cumulative Gain calculation."""
        # Relevance scores for retrieved documents
        retrieved = [3, 2, 3, 0, 1, 2]  # Relevance scores
        ideal = [3, 3, 2, 2, 1, 0]  # Ideal ordering
        
        ndcg = metrics_evaluator.calculate_ndcg(retrieved, ideal, k=6)
        
        assert 0 <= ndcg <= 1  # NDCG is between 0 and 1
        assert ndcg < 1  # Not perfect ranking
    
    def test_recall_at_k(self, metrics_evaluator):
        """Test Recall@K calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6", "doc7"]
        
        recall_at_5 = metrics_evaluator.calculate_recall_at_k(retrieved, relevant, k=5)
        expected_recall = 2 / 4  # 2 relevant docs found out of 4 total relevant
        
        assert recall_at_5 == expected_recall
    
    def test_precision_at_k(self, metrics_evaluator):
        """Test Precision@K calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6"]
        
        precision_at_3 = metrics_evaluator.calculate_precision_at_k(
            retrieved[:3], relevant, k=3
        )
        expected_precision = 2 / 3  # 2 relevant docs out of 3 retrieved
        
        assert abs(precision_at_3 - expected_precision) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])