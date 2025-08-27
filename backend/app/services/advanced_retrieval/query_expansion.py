"""
Query Expansion module for enhancing search queries.
Implements T5-based generation, BERT-based synonym expansion, and pseudo-relevance feedback.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import re
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModel
)
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class QueryExpansion:
    """
    Implements multiple query expansion techniques to improve retrieval recall.
    
    Methods:
    - T5-based query generation
    - BERT-based synonym expansion
    - Pseudo-relevance feedback
    - Combined expansion strategies
    """
    
    def __init__(
        self,
        t5_model_name: str = "t5-small",
        bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize query expansion models.
        
        Args:
            t5_model_name: Name of T5 model for query generation
            bert_model_name: Name of BERT model for embeddings
            device: Device to run models on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models lazily to save memory
        self._t5_model = None
        self._t5_tokenizer = None
        self._bert_model = None
        self.t5_model_name = t5_model_name
        self.bert_model_name = bert_model_name
        
        logger.info(f"QueryExpansion initialized with device: {self.device}")
    
    @property
    def t5_model(self):
        """Lazy load T5 model."""
        if self._t5_model is None:
            self._t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
            self._t5_model = T5ForConditionalGeneration.from_pretrained(
                self.t5_model_name
            ).to(self.device)
            self._t5_model.eval()
        return self._t5_model
    
    @property
    def t5_tokenizer(self):
        """Get T5 tokenizer."""
        if self._t5_tokenizer is None:
            self._t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
        return self._t5_tokenizer
    
    @property
    def bert_model(self):
        """Lazy load BERT model."""
        if self._bert_model is None:
            self._bert_model = SentenceTransformer(self.bert_model_name)
        return self._bert_model
    
    async def expand_with_t5(
        self,
        query: str,
        num_expansions: int = 3,
        max_length: int = 64,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate query expansions using T5 model.
        
        Args:
            query: Original query to expand
            num_expansions: Number of expansions to generate
            max_length: Maximum length of generated queries
            temperature: Sampling temperature for generation
            
        Returns:
            List of expanded queries including original
        """
        try:
            # Prepare prompt for T5
            prompt = f"expand query: {query}"
            
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            expansions = await loop.run_in_executor(
                None,
                self._generate_t5_expansions,
                prompt,
                num_expansions,
                max_length,
                temperature
            )
            
            # Include original query
            result = [query] + expansions
            
            # Remove duplicates while preserving order
            seen = set()
            unique_result = []
            for q in result:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_result.append(q)
            
            logger.debug(f"T5 expansion for '{query}': {unique_result}")
            return unique_result
            
        except Exception as e:
            logger.error(f"T5 expansion failed: {str(e)}")
            return [query]  # Return original query on failure
    
    def _generate_t5_expansions(
        self,
        prompt: str,
        num_expansions: int,
        max_length: int,
        temperature: float
    ) -> List[str]:
        """Helper method to generate T5 expansions (runs in executor)."""
        inputs = self.t5_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=num_expansions,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_beams=2,
                early_stopping=True
            )
        
        expansions = [
            self.t5_tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return expansions
    
    async def expand_with_bert(
        self,
        query: str,
        max_synonyms: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """
        Expand query using BERT embeddings and WordNet synonyms.
        
        Args:
            query: Original query to expand
            max_synonyms: Maximum synonyms per word
            similarity_threshold: Minimum similarity for synonym inclusion
            
        Returns:
            List of expanded queries with synonyms
        """
        try:
            # Tokenize query
            tokens = word_tokenize(query.lower())
            
            # Get POS tags for better synonym matching
            pos_tags = nltk.pos_tag(tokens)
            
            # Find synonyms for each word
            synonyms_map = {}
            for word, pos in pos_tags:
                # Skip stop words and short words
                if len(word) <= 2:
                    continue
                
                # Get WordNet synonyms
                synonyms = self._get_wordnet_synonyms(word, pos, max_synonyms)
                if synonyms:
                    synonyms_map[word] = synonyms
            
            # Generate expanded queries
            expanded_queries = [query]
            
            # Create variations with synonyms
            for word, synonyms in synonyms_map.items():
                for synonym in synonyms[:max_synonyms]:
                    # Replace word with synonym in query
                    expanded = query.lower().replace(word, synonym)
                    if expanded != query.lower():
                        expanded_queries.append(expanded)
            
            # Also create a query with all main synonyms
            if synonyms_map:
                all_terms = set(tokens)
                for syns in synonyms_map.values():
                    all_terms.update(syns[:1])  # Add top synonym for each word
                expanded_queries.append(' '.join(all_terms))
            
            # Remove duplicates
            unique_queries = list(dict.fromkeys(expanded_queries))
            
            logger.debug(f"BERT/WordNet expansion for '{query}': {unique_queries}")
            return unique_queries
            
        except Exception as e:
            logger.error(f"BERT expansion failed: {str(e)}")
            return [query]
    
    def _get_wordnet_synonyms(
        self,
        word: str,
        pos: str,
        max_synonyms: int
    ) -> List[str]:
        """Get WordNet synonyms for a word based on POS tag."""
        # Map POS tags to WordNet POS
        pos_map = {
            'NN': wordnet.NOUN,
            'VB': wordnet.VERB,
            'JJ': wordnet.ADJ,
            'RB': wordnet.ADV
        }
        
        wordnet_pos = pos_map.get(pos[:2], wordnet.NOUN)
        
        synonyms = set()
        for synset in wordnet.synsets(word, pos=wordnet_pos)[:3]:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) > 2:
                    synonyms.add(synonym)
        
        return list(synonyms)[:max_synonyms]
    
    async def pseudo_relevance_expansion(
        self,
        query: str,
        top_docs: List[Dict[str, Any]],
        num_terms: int = 5,
        min_doc_freq: int = 2
    ) -> List[str]:
        """
        Expand query using terms from top retrieved documents (pseudo-relevance feedback).
        
        Args:
            query: Original query
            top_docs: Top retrieved documents with 'content' field
            num_terms: Number of expansion terms to add
            min_doc_freq: Minimum document frequency for term inclusion
            
        Returns:
            List of expanded queries
        """
        try:
            if not top_docs:
                return [query]
            
            # Extract content from documents
            doc_contents = [doc.get('content', '') for doc in top_docs[:5]]
            
            # Tokenize all documents
            all_tokens = []
            query_tokens = set(word_tokenize(query.lower()))
            
            for content in doc_contents:
                tokens = word_tokenize(content.lower())
                # Filter tokens
                tokens = [
                    t for t in tokens 
                    if len(t) > 3 and t.isalpha() and t not in query_tokens
                ]
                all_tokens.extend(tokens)
            
            # Count term frequencies
            term_freq = Counter(all_tokens)
            
            # Filter by document frequency
            term_doc_freq = {}
            for term in term_freq:
                doc_count = sum(1 for content in doc_contents if term in content.lower())
                if doc_count >= min_doc_freq:
                    term_doc_freq[term] = term_freq[term]
            
            # Get top terms
            top_terms = sorted(
                term_doc_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:num_terms]
            
            # Create expanded queries
            expanded_queries = [query]
            
            if top_terms:
                # Add individual terms
                for term, _ in top_terms[:3]:
                    expanded_queries.append(f"{query} {term}")
                
                # Add all terms together
                all_expansion_terms = ' '.join([term for term, _ in top_terms])
                expanded_queries.append(f"{query} {all_expansion_terms}")
            
            logger.debug(f"Pseudo-relevance expansion for '{query}': {expanded_queries}")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Pseudo-relevance expansion failed: {str(e)}")
            return [query]
    
    async def combine_expansions(
        self,
        query: str,
        methods: List[str] = ['t5', 'bert', 'pseudo'],
        top_docs: Optional[List[Dict[str, Any]]] = None,
        max_expansions: int = 10
    ) -> List[str]:
        """
        Combine multiple expansion methods for comprehensive query expansion.
        
        Args:
            query: Original query
            methods: List of expansion methods to use
            top_docs: Documents for pseudo-relevance feedback
            max_expansions: Maximum number of expansions to return
            
        Returns:
            Combined list of expanded queries
        """
        all_expansions = set([query])
        tasks = []
        
        if 't5' in methods:
            tasks.append(self.expand_with_t5(query))
        
        if 'bert' in methods:
            tasks.append(self.expand_with_bert(query))
        
        if 'pseudo' in methods and top_docs:
            tasks.append(self.pseudo_relevance_expansion(query, top_docs))
        
        # Run all expansions concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_expansions.update(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Expansion method failed: {str(result)}")
        
        # Convert to list and limit size
        expanded_queries = list(all_expansions)[:max_expansions]
        
        # Ensure original query is first
        if query in expanded_queries:
            expanded_queries.remove(query)
        expanded_queries.insert(0, query)
        
        logger.info(f"Combined expansion generated {len(expanded_queries)} queries from '{query}'")
        return expanded_queries
    
    def get_expansion_embeddings(
        self,
        queries: List[str]
    ) -> np.ndarray:
        """
        Get embeddings for expanded queries for similarity comparison.
        
        Args:
            queries: List of queries to embed
            
        Returns:
            Array of query embeddings
        """
        embeddings = self.bert_model.encode(
            queries,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embeddings
    
    def filter_similar_expansions(
        self,
        expansions: List[str],
        similarity_threshold: float = 0.9
    ) -> List[str]:
        """
        Filter out highly similar query expansions to reduce redundancy.
        
        Args:
            expansions: List of expanded queries
            similarity_threshold: Maximum similarity to consider queries different
            
        Returns:
            Filtered list of diverse expansions
        """
        if len(expansions) <= 1:
            return expansions
        
        # Get embeddings
        embeddings = self.get_expansion_embeddings(expansions)
        
        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)
        
        # Keep track of which queries to keep
        keep_indices = [0]  # Always keep the first (original) query
        
        for i in range(1, len(expansions)):
            # Check similarity with all kept queries
            is_different = True
            for j in keep_indices:
                if similarities[i, j] > similarity_threshold:
                    is_different = False
                    break
            
            if is_different:
                keep_indices.append(i)
        
        filtered = [expansions[i] for i in keep_indices]
        
        logger.debug(f"Filtered {len(expansions)} expansions to {len(filtered)} diverse queries")
        return filtered