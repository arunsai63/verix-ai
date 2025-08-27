"""
Semantic Chunking implementation using Sentence-BERT for intelligent text segmentation.
Creates semantically coherent chunks based on meaning rather than fixed size.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass
from itertools import groupby

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class ChunkConfig:
    """Configuration for semantic chunking."""
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    similarity_threshold: float = 0.75
    overlap_size: int = 50
    preserve_sentences: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticChunker:
    """
    Semantic chunking based on sentence embeddings and similarity.
    
    Creates chunks by identifying semantic boundaries where the meaning
    changes significantly between sentences.
    """
    
    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            config: Chunking configuration
            embedding_model: Pre-loaded embedding model
        """
        self.config = config or ChunkConfig()
        self.min_chunk_size = self.config.min_chunk_size
        self.max_chunk_size = self.config.max_chunk_size
        self.similarity_threshold = self.config.similarity_threshold
        self.model_name = self.config.model_name
        
        # Initialize embedding model
        self._embedding_model = embedding_model
        if self._embedding_model is None:
            self._load_embedding_model()
        
        logger.info(f"SemanticChunker initialized with threshold: {self.similarity_threshold}")
    
    def _load_embedding_model(self):
        """Load the sentence embedding model."""
        try:
            self._embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Get the embedding model."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences and clean
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def find_semantic_boundaries(
        self,
        sentences: List[str],
        threshold: Optional[float] = None
    ) -> List[int]:
        """
        Find semantic boundaries between sentences.
        
        Args:
            sentences: List of sentences
            threshold: Similarity threshold for boundary detection
            
        Returns:
            List of boundary indices
        """
        if len(sentences) <= 1:
            return []
        
        threshold = threshold or self.similarity_threshold
        
        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Calculate similarities between consecutive sentences
        boundaries = []
        for i in range(len(sentences) - 1):
            similarity = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            
            # Mark boundary if similarity is below threshold
            if similarity < threshold:
                boundaries.append(i + 1)  # Boundary after sentence i
        
        return boundaries
    
    def chunk_document(
        self,
        text: str,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document based on semantic boundaries.
        
        Args:
            text: Document text
            min_size: Minimum chunk size
            max_size: Maximum chunk size
            similarity_threshold: Threshold for boundaries
            
        Returns:
            List of chunks with metadata
        """
        min_size = min_size or self.min_chunk_size
        max_size = max_size or self.max_chunk_size
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        
        # Find semantic boundaries
        boundaries = self.find_semantic_boundaries(sentences, similarity_threshold)
        
        # Add start and end boundaries
        boundaries = [0] + boundaries + [len(sentences)]
        
        # Create chunks based on boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Check size constraints
            if len(chunk_text) < min_size and i < len(boundaries) - 2:
                # Merge with next chunk if too small
                continue
            
            if len(chunk_text) > max_size:
                # Split large chunk further
                sub_chunks = self._split_large_chunk(
                    chunk_sentences,
                    max_size
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'chunk_index': len(chunks),
                        'start_sentence': start_idx,
                        'end_sentence': end_idx,
                        'num_sentences': len(chunk_sentences),
                        'chunk_type': 'semantic'
                    }
                })
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks, min_size, max_size)
        
        return chunks
    
    def _split_large_chunk(
        self,
        sentences: List[str],
        max_size: int
    ) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones."""
        sub_chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_size and current_chunk:
                # Create chunk
                sub_chunks.append({
                    'content': ' '.join(current_chunk),
                    'metadata': {
                        'chunk_type': 'semantic_split'
                    }
                })
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
        
        # Add last chunk
        if current_chunk:
            sub_chunks.append({
                'content': ' '.join(current_chunk),
                'metadata': {
                    'chunk_type': 'semantic_split'
                }
            })
        
        return sub_chunks
    
    def _post_process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_size: int,
        max_size: int
    ) -> List[Dict[str, Any]]:
        """Post-process chunks to ensure size constraints."""
        processed = []
        buffer = None
        
        for chunk in chunks:
            content = chunk['content']
            
            # Handle small chunks
            if len(content) < min_size:
                if buffer:
                    # Merge with buffer
                    buffer['content'] += ' ' + content
                else:
                    # Start new buffer
                    buffer = chunk
                
                # Check if buffer is large enough
                if buffer and len(buffer['content']) >= min_size:
                    processed.append(buffer)
                    buffer = None
            else:
                # Add buffer if exists
                if buffer:
                    if len(buffer['content']) + len(content) <= max_size:
                        # Merge buffer with current chunk
                        chunk['content'] = buffer['content'] + ' ' + content
                    else:
                        # Add buffer as separate chunk
                        processed.append(buffer)
                    buffer = None
                
                processed.append(chunk)
        
        # Add remaining buffer
        if buffer:
            processed.append(buffer)
        
        # Update indices
        for i, chunk in enumerate(processed):
            chunk['metadata']['chunk_index'] = i
        
        return processed
    
    def calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score for a text chunk.
        
        Args:
            text: Text chunk
            
        Returns:
            Coherence score (0-1)
        """
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 1:
            return 1.0
        
        # Get embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    [embeddings[i]],
                    [embeddings[j]]
                )[0][0]
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        return float(np.mean(similarities))
    
    def chunk_by_topics(
        self,
        text: str,
        num_topics: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Chunk document based on topic modeling.
        
        Args:
            text: Document text
            num_topics: Number of topics to identify
            
        Returns:
            List of topic-based chunks
        """
        try:
            from bertopic import BERTopic
            
            sentences = self.split_into_sentences(text)
            if len(sentences) < num_topics:
                # Fall back to semantic chunking
                return self.chunk_document(text)
            
            # Create topic model
            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                nr_topics=num_topics,
                verbose=False
            )
            
            # Fit model and get topics
            topics, _ = topic_model.fit_transform(sentences)
            
            # Group sentences by topic
            chunks = []
            for topic_id in set(topics):
                if topic_id == -1:  # Outlier topic
                    continue
                
                topic_sentences = [
                    sent for sent, topic in zip(sentences, topics)
                    if topic == topic_id
                ]
                
                if topic_sentences:
                    chunks.append({
                        'content': ' '.join(topic_sentences),
                        'metadata': {
                            'chunk_index': len(chunks),
                            'topic_id': int(topic_id),
                            'num_sentences': len(topic_sentences),
                            'chunk_type': 'topic'
                        }
                    })
            
            # Handle outliers
            outlier_sentences = [
                sent for sent, topic in zip(sentences, topics)
                if topic == -1
            ]
            if outlier_sentences:
                chunks.append({
                    'content': ' '.join(outlier_sentences),
                    'metadata': {
                        'chunk_index': len(chunks),
                        'topic_id': -1,
                        'num_sentences': len(outlier_sentences),
                        'chunk_type': 'topic_outlier'
                    }
                })
            
            return chunks
            
        except ImportError:
            logger.warning("BERTopic not available, falling back to semantic chunking")
            return self.chunk_document(text)
    
    def adaptive_chunk(
        self,
        text: str,
        base_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Adaptively chunk based on content characteristics.
        
        Args:
            text: Document text
            base_size: Base chunk size
            
        Returns:
            List of adaptive chunks
        """
        # Analyze text complexity
        sentences = self.split_into_sentences(text)
        
        # Calculate sentence complexities
        complexities = []
        for sentence in sentences:
            # Simple complexity: word count and average word length
            words = sentence.split()
            if words:
                complexity = len(words) * (sum(len(w) for w in words) / len(words))
            else:
                complexity = 0
            complexities.append(complexity)
        
        # Group similar complexity sentences
        chunks = []
        current_chunk = []
        current_complexity = 0
        
        for sentence, complexity in zip(sentences, complexities):
            if not current_chunk:
                current_chunk = [sentence]
                current_complexity = complexity
            elif abs(complexity - current_complexity) / (current_complexity + 1) < 0.5:
                # Similar complexity, add to current chunk
                current_chunk.append(sentence)
            else:
                # Different complexity, start new chunk
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'metadata': {
                        'chunk_index': len(chunks),
                        'avg_complexity': current_complexity / len(current_chunk),
                        'chunk_type': 'adaptive'
                    }
                })
                current_chunk = [sentence]
                current_complexity = complexity
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'metadata': {
                    'chunk_index': len(chunks),
                    'avg_complexity': current_complexity / len(current_chunk),
                    'chunk_type': 'adaptive'
                }
            })
        
        return chunks