"""
BM25 (Okapi BM25) sparse retrieval implementation.
Provides efficient keyword-based document retrieval using term frequency and inverse document frequency.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')  
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class BM25Retriever:
    """
    Implements Okapi BM25 ranking function for document retrieval.
    
    BM25 is a probabilistic retrieval model that ranks documents based on:
    - Term frequency (TF)
    - Inverse document frequency (IDF)  
    - Document length normalization
    
    Parameters:
        k1: Controls term frequency saturation (default: 1.2)
        b: Controls length normalization (0=no normalization, 1=full normalization) (default: 0.75)
        epsilon: Floor value for IDF to prevent negative values (default: 0.25)
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        """
        Initialize BM25 retriever with tuning parameters.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
            epsilon: IDF floor value to prevent negative scores
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Document processing
        self.corpus = []
        self.corpus_size = 0
        self.doc_len = []
        self.avgdl = 0
        
        # Indexing structures
        self.inverted_index = {}
        self.doc_freqs = {}
        self.idf = {}
        
        # Text processing
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"BM25 Retriever initialized with k1={k1}, b={b}, epsilon={epsilon}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing and search.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Stem tokens
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def fit(self, documents: List[str]) -> 'BM25Retriever':
        """
        Fit BM25 model on a corpus of documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Self for method chaining
        """
        self.corpus_size = len(documents)
        self.corpus = []
        self.doc_len = []
        
        # Process documents and build inverted index
        for doc_id, doc in enumerate(documents):
            processed_doc = self._preprocess_text(doc)
            self.corpus.append(processed_doc)
            self.doc_len.append(len(processed_doc))
            
            # Update inverted index
            for term in set(processed_doc):
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(doc_id)
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate document frequencies and IDF scores
        self._calculate_idf()
        
        logger.info(f"BM25 fitted on {self.corpus_size} documents, "
                   f"vocabulary size: {len(self.inverted_index)}")
        
        return self
    
    def _calculate_idf(self):
        """Calculate IDF scores for all terms in the corpus."""
        for term, doc_ids in self.inverted_index.items():
            df = len(doc_ids)
            self.doc_freqs[term] = df
            # IDF formula with epsilon floor
            idf_score = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = max(idf_score, self.epsilon)
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for all documents given a query.
        
        Args:
            query: Search query string
            
        Returns:
            Array of BM25 scores for each document
        """
        query_terms = self._preprocess_text(query)
        scores = np.zeros(self.corpus_size)
        
        # Calculate term frequencies in query
        query_term_freqs = Counter(query_terms)
        
        for term, query_tf in query_term_freqs.items():
            if term not in self.inverted_index:
                continue
                
            idf_score = self.idf[term]
            
            # Calculate BM25 score for this term across all documents
            for doc_id in self.inverted_index[term]:
                doc_tf = self.corpus[doc_id].count(term)
                doc_length = self.doc_len[doc_id]
                
                # BM25 scoring formula
                numerator = idf_score * doc_tf * (self.k1 + 1)
                denominator = doc_tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                
                scores[doc_id] += numerator / denominator
        
        return scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-k documents for a query based on BM25 scores.
        
        Args:
            query: Search query string
            k: Number of top documents to return
            
        Returns:
            List of dictionaries with document index, score, and rank
        """
        scores = self.get_scores(query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append({
                    'index': int(idx),
                    'score': float(scores[idx]),
                    'rank': rank
                })
        
        return results
    
    def update_corpus(self, new_documents: List[str]) -> 'BM25Retriever':
        """
        Incrementally update the corpus with new documents.
        
        Args:
            new_documents: List of new document strings to add
            
        Returns:
            Self for method chaining
        """
        start_id = self.corpus_size
        
        for doc_id, doc in enumerate(new_documents, start=start_id):
            processed_doc = self._preprocess_text(doc)
            self.corpus.append(processed_doc)
            self.doc_len.append(len(processed_doc))
            
            # Update inverted index
            for term in set(processed_doc):
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(doc_id)
        
        self.corpus_size = len(self.corpus)
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Recalculate IDF scores
        self._calculate_idf()
        
        logger.info(f"Corpus updated with {len(new_documents)} new documents. "
                   f"Total size: {self.corpus_size}")
        
        return self
    
    def save_index(self, path: str):
        """
        Save the BM25 index to disk for later use.
        
        Args:
            path: File path to save the index
        """
        import pickle
        
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon,
            'corpus': self.corpus,
            'corpus_size': self.corpus_size,
            'doc_len': self.doc_len,
            'avgdl': self.avgdl,
            'inverted_index': self.inverted_index,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load_index(self, path: str) -> 'BM25Retriever':
        """
        Load a previously saved BM25 index from disk.
        
        Args:
            path: File path to load the index from
            
        Returns:
            Self for method chaining
        """
        import pickle
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.k1 = index_data['k1']
        self.b = index_data['b']
        self.epsilon = index_data['epsilon']
        self.corpus = index_data['corpus']
        self.corpus_size = index_data['corpus_size']
        self.doc_len = index_data['doc_len']
        self.avgdl = index_data['avgdl']
        self.inverted_index = index_data['inverted_index']
        self.doc_freqs = index_data['doc_freqs']
        self.idf = index_data['idf']
        
        logger.info(f"BM25 index loaded from {path}")
        
        return self
    
    def get_term_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed terms.
        
        Returns:
            Dictionary with vocabulary size, average document length, etc.
        """
        return {
            'vocabulary_size': len(self.inverted_index),
            'corpus_size': self.corpus_size,
            'average_doc_length': self.avgdl,
            'total_terms': sum(self.doc_len),
            'most_common_terms': sorted(
                self.doc_freqs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }