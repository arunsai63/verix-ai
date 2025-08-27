"""
Dynamic Chunking implementation with adaptive sizing based on content complexity.
Adjusts chunk sizes dynamically based on content characteristics.
"""

import logging
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DynamicChunkConfig:
    """Configuration for dynamic chunking."""
    base_size: int = 500
    min_size: int = 100
    max_size: int = 2000
    complexity_weight: float = 0.5
    preserve_format: bool = True
    overlap_ratio: float = 0.1


class DynamicChunker:
    """
    Dynamic chunking that adapts chunk size based on content complexity.
    
    Adjusts chunk boundaries based on:
    - Content complexity (vocabulary, sentence structure)
    - Information density
    - Format requirements (code, lists, tables)
    """
    
    def __init__(self, config: Optional[DynamicChunkConfig] = None):
        """
        Initialize dynamic chunker.
        
        Args:
            config: Dynamic chunking configuration
        """
        self.config = config or DynamicChunkConfig()
        self.base_size = self.config.base_size
        self.min_size = self.config.min_size
        self.max_size = self.config.max_size
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            logger.warning("Tiktoken not available, using word-based counting")
        
        logger.info("DynamicChunker initialized")
    
    def analyze_complexity(self, text: str) -> float:
        """
        Analyze text complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complexity score (0-1)
        """
        if not text:
            return 0.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate various complexity metrics
        metrics = []
        
        # 1. Average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        sentence_complexity = min(avg_sentence_length / 30, 1.0)  # Normalize to 0-1
        metrics.append(sentence_complexity)
        
        # 2. Vocabulary diversity (unique words / total words)
        words = text.lower().split()
        if words:
            vocab_diversity = len(set(words)) / len(words)
            metrics.append(vocab_diversity)
        
        # 3. Average word length
        if words:
            avg_word_length = np.mean([len(w) for w in words])
            word_complexity = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
            metrics.append(word_complexity)
        
        # 4. Technical indicator (presence of technical terms)
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+\.\d+\b',    # Numbers with decimals
            r'\w+\(\)',         # Function calls
            r'[{}\[\]]',        # Code-like brackets
        ]
        technical_score = sum(
            1 for pattern in technical_patterns 
            if re.search(pattern, text)
        ) / len(technical_patterns)
        metrics.append(technical_score)
        
        # 5. Nested structure (parentheses, commas)
        nesting_chars = text.count('(') + text.count(',') + text.count(';')
        nesting_score = min(nesting_chars / len(text) * 10, 1.0)
        metrics.append(nesting_score)
        
        # Combine metrics
        complexity = np.mean(metrics)
        
        return float(complexity)
    
    def calculate_chunk_size(
        self,
        text: str,
        base_size: Optional[int] = None
    ) -> int:
        """
        Calculate optimal chunk size based on content.
        
        Args:
            text: Text to chunk
            base_size: Base chunk size
            
        Returns:
            Calculated chunk size
        """
        base_size = base_size or self.base_size
        
        # Analyze complexity
        complexity = self.analyze_complexity(text)
        
        # Adjust size based on complexity
        # Higher complexity -> smaller chunks
        size_multiplier = 1.5 - complexity * self.config.complexity_weight
        
        chunk_size = int(base_size * size_multiplier)
        
        # Apply bounds
        chunk_size = max(self.min_size, min(self.max_size, chunk_size))
        
        return chunk_size
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            model: Model to use for tokenization
            
        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to word count estimation
            return len(text.split())
    
    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-3.5-turbo",
        preserve_sentences: bool = True
    ) -> List[str]:
        """
        Split text by token count.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            model: Model for tokenization
            preserve_sentences: Try to preserve sentence boundaries
            
        Returns:
            List of text chunks
        """
        if preserve_sentences:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence, model)
                
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        else:
            # Hard split by tokens
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                chunks = []
                for i in range(0, len(tokens), max_tokens):
                    chunk_tokens = tokens[i:i+max_tokens]
                    chunks.append(self.tokenizer.decode(chunk_tokens))
                return chunks
            else:
                # Fallback to character-based splitting
                words = text.split()
                chunks = []
                for i in range(0, len(words), max_tokens):
                    chunks.append(' '.join(words[i:i+max_tokens]))
                return chunks
    
    def chunk_with_overlap(
        self,
        text: str,
        chunk_size: int,
        overlap_ratio: Optional[float] = None,
        preserve_sentences: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Create chunks with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap_ratio: Overlap ratio (0-1)
            preserve_sentences: Preserve sentence boundaries
            
        Returns:
            List of chunks with metadata
        """
        overlap_ratio = overlap_ratio or self.config.overlap_ratio
        overlap_size = int(chunk_size * overlap_ratio)
        
        if preserve_sentences:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            i = 0
            
            while i < len(sentences):
                # Build chunk
                current_chunk = []
                current_size = 0
                j = i
                
                while j < len(sentences) and current_size < chunk_size:
                    sentence = sentences[j]
                    current_chunk.append(sentence)
                    current_size += len(sentence) + 1
                    j += 1
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'start_sentence': i,
                            'end_sentence': j,
                            'has_overlap': i > 0
                        }
                    })
                
                # Move with overlap
                if j < len(sentences):
                    # Find overlap point
                    overlap_sentences = max(1, len(current_chunk) // 3)
                    i = j - overlap_sentences
                else:
                    break
            
            return chunks
        else:
            # Character-based overlap
            chunks = []
            i = 0
            
            while i < len(text):
                end = min(i + chunk_size, len(text))
                chunk_text = text[i:end]
                
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'start_pos': i,
                        'end_pos': end,
                        'has_overlap': i > 0
                    }
                })
                
                if end < len(text):
                    i = end - overlap_size
                else:
                    break
            
            return chunks
    
    def chunk_document(
        self,
        text: str,
        base_size: Optional[int] = None,
        preserve_format: Optional[bool] = None,
        format_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Dynamically chunk document based on content.
        
        Args:
            text: Document text
            base_size: Base chunk size
            preserve_format: Preserve special formats
            format_type: Type of format to preserve
            
        Returns:
            List of chunks
        """
        base_size = base_size or self.base_size
        preserve_format = preserve_format if preserve_format is not None else self.config.preserve_format
        
        # Detect format if not specified
        if not format_type:
            format_type = self._detect_format(text)
        
        if preserve_format and format_type:
            return self._chunk_by_format(text, format_type, base_size)
        else:
            return self._adaptive_chunk(text, base_size)
    
    def _detect_format(self, text: str) -> Optional[str]:
        """Detect special format in text."""
        # Check for code
        if re.search(r'```|def |class |function |import ', text):
            return 'code'
        
        # Check for lists
        if re.search(r'^\s*[-*•]\s+', text, re.MULTILINE):
            return 'list'
        
        # Check for tables
        if re.search(r'\|.*\|', text):
            return 'table'
        
        return None
    
    def _chunk_by_format(
        self,
        text: str,
        format_type: str,
        base_size: int
    ) -> List[Dict[str, Any]]:
        """Chunk while preserving format."""
        chunks = []
        
        if format_type == 'code':
            # Split by code blocks
            parts = re.split(r'(```[\s\S]*?```)', text)
            
            for i, part in enumerate(parts):
                if part.startswith('```'):
                    # Code block - keep together if possible
                    chunks.append({
                        'content': part,
                        'metadata': {
                            'format': 'code_block',
                            'chunk_index': len(chunks)
                        }
                    })
                else:
                    # Regular text - chunk normally
                    if part.strip():
                        sub_chunks = self._adaptive_chunk(part, base_size)
                        chunks.extend(sub_chunks)
        
        elif format_type == 'list':
            # Split by list items
            lines = text.split('\n')
            current_list = []
            current_size = 0
            
            for line in lines:
                if re.match(r'^\s*[-*•]\s+', line):
                    # List item
                    if current_size + len(line) > base_size and current_list:
                        chunks.append({
                            'content': '\n'.join(current_list),
                            'metadata': {
                                'format': 'list',
                                'chunk_index': len(chunks)
                            }
                        })
                        current_list = [line]
                        current_size = len(line)
                    else:
                        current_list.append(line)
                        current_size += len(line)
                else:
                    # Non-list content
                    if current_list:
                        chunks.append({
                            'content': '\n'.join(current_list),
                            'metadata': {
                                'format': 'list',
                                'chunk_index': len(chunks)
                            }
                        })
                        current_list = []
                        current_size = 0
                    
                    if line.strip():
                        chunks.append({
                            'content': line,
                            'metadata': {
                                'format': 'text',
                                'chunk_index': len(chunks)
                            }
                        })
            
            if current_list:
                chunks.append({
                    'content': '\n'.join(current_list),
                    'metadata': {
                        'format': 'list',
                        'chunk_index': len(chunks)
                    }
                })
        
        else:
            # Default format preservation
            chunks = self._adaptive_chunk(text, base_size)
        
        return chunks
    
    def _adaptive_chunk(
        self,
        text: str,
        base_size: int
    ) -> List[Dict[str, Any]]:
        """Adaptively chunk based on content complexity."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Calculate size for this paragraph
            para_complexity = self.analyze_complexity(paragraph)
            para_size = self.calculate_chunk_size(paragraph, base_size)
            
            if current_size + len(paragraph) > para_size and current_chunk:
                # Create chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'chunk_index': len(chunks),
                        'avg_complexity': self.analyze_complexity(chunk_content),
                        'chunk_type': 'adaptive'
                    }
                })
                current_chunk = [paragraph]
                current_size = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_size += len(paragraph) + 2  # +2 for \n\n
        
        # Add last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'chunk_index': len(chunks),
                    'avg_complexity': self.analyze_complexity(chunk_content),
                    'chunk_type': 'adaptive'
                }
            })
        
        return chunks
    
    def adaptive_chunk(
        self,
        text: str,
        base_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Public interface for adaptive chunking.
        
        Args:
            text: Text to chunk
            base_size: Base chunk size
            
        Returns:
            List of adaptive chunks
        """
        return self._adaptive_chunk(text, base_size or self.base_size)
    
    def calculate_statistics(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_size': 0,
                'min_size': 0,
                'max_size': 0,
                'size_std': 0
            }
        
        sizes = [len(c['content']) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_size': np.mean(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'size_std': np.std(sizes),
            'total_chars': sum(sizes)
        }