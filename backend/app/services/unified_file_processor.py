import os
import hashlib
import logging
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from markitdown import MarkItDown
import pypdf
import io

logger = logging.getLogger(__name__)


class UnifiedFileProcessor:
    """Unified processor for all documents with streaming and optimization."""
    
    def __init__(self):
        self.converter = MarkItDown()
        self.max_file_size = 500 * 1024 * 1024  # 500MB max
        self.chunk_size = 1000  # Standard chunk size
        self.chunk_overlap = 200  # Standard overlap
        self.large_file_threshold = 50 * 1024 * 1024  # 50MB threshold
        self.max_chunks_per_batch = 100  # Process in batches
        
    def _get_optimal_chunk_size(self, file_size: int) -> tuple[int, int]:
        """Get optimal chunk size and overlap based on file size."""
        if file_size < 1 * 1024 * 1024:  # < 1MB
            return 1000, 200
        elif file_size < 10 * 1024 * 1024:  # < 10MB
            return 800, 150
        elif file_size < 50 * 1024 * 1024:  # < 50MB
            return 600, 100
        else:  # >= 50MB
            return 500, 50
    
    def process_pdf(
        self,
        file_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process PDF files with streaming and memory optimization for all sizes.
        
        Args:
            file_path: Path to the PDF file
            dataset_name: Name of the dataset
            metadata: Optional metadata
            progress_callback: Optional callback for progress updates
        
        Yields:
            Batches of processed chunks
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.2f}MB > {self.max_file_size / 1024 / 1024:.2f}MB max")
        
        logger.info(f"Processing PDF: {file_path.name} ({file_size / 1024 / 1024:.2f}MB)")
        
        # Get optimal chunk size based on file size
        chunk_size, chunk_overlap = self._get_optimal_chunk_size(file_size)
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash_streaming(file_path)
        
        base_metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_extension": ".pdf",
            "file_size": file_size,
            "file_size_mb": file_size / 1024 / 1024,
            "file_hash": file_hash,
            "dataset_name": dataset_name,
            "processed_at": datetime.utcnow().isoformat(),
            "source_type": "pdf",
            "large_file": True,
            **(metadata or {})
        }
        
        # Process PDF page by page
        total_pages = 0
        total_chunks = 0
        chunks_batch = []
        
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = pypdf.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF has {total_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    # Extract text from page
                    try:
                        page_text = page.extract_text()
                        
                        if not page_text or len(page_text.strip()) < 10:
                            continue
                        
                        # Create chunks from page text with optimal size
                        page_chunks = self._chunk_text(
                            page_text,
                            {
                                **base_metadata,
                                "page_number": page_num + 1,
                                "total_pages": total_pages
                            },
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        chunks_batch.extend(page_chunks)
                        total_chunks += len(page_chunks)
                        
                        # Yield batch when it reaches the limit
                        if len(chunks_batch) >= self.max_chunks_per_batch:
                            yield {
                                "chunks": chunks_batch,
                                "metadata": base_metadata,
                                "batch_size": len(chunks_batch),
                                "total_chunks_so_far": total_chunks
                            }
                            chunks_batch = []
                        
                        # Report progress
                        if progress_callback and page_num % 10 == 0:
                            progress = ((page_num + 1) / total_pages) * 100
                            progress_callback({
                                "progress": progress,
                                "pages_processed": page_num + 1,
                                "total_pages": total_pages,
                                "chunks_created": total_chunks
                            })
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                
                # Yield remaining chunks
                if chunks_batch:
                    yield {
                        "chunks": chunks_batch,
                        "metadata": base_metadata,
                        "batch_size": len(chunks_batch),
                        "total_chunks_so_far": total_chunks
                    }
                
                logger.info(f"Successfully processed {total_pages} pages, created {total_chunks} chunks")
                
        except Exception as e:
            logger.error(f"Error processing large PDF {file_path}: {str(e)}")
            raise
    
    def process_file_generic(
        self,
        file_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_temp_file: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process non-PDF files using MarkItDown with memory optimization for all sizes.
        
        Args:
            file_path: Path to the file
            dataset_name: Name of the dataset
            metadata: Optional metadata
            use_temp_file: Use temporary file for processing
        
        Yields:
            Batches of processed chunks
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.2f}MB")
        
        logger.info(f"Processing file: {file_path.name} ({file_size / 1024 / 1024:.2f}MB)")
        
        # Get optimal chunk size based on file size
        chunk_size, chunk_overlap = self._get_optimal_chunk_size(file_size)
        
        # For large files, use a temporary copy to avoid locking
        if use_temp_file and file_size > self.large_file_threshold:
            with tempfile.NamedTemporaryFile(suffix=file_path.suffix, delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
                try:
                    shutil.copy2(file_path, temp_path)
                    result = self.converter.convert(str(temp_path))
                finally:
                    temp_path.unlink(missing_ok=True)
        else:
            result = self.converter.convert(str(file_path))
        
        content = result.text_content
        
        # Calculate hash
        file_hash = self._calculate_file_hash_streaming(file_path)
        
        base_metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_extension": file_path.suffix.lower(),
            "file_size": file_size,
            "file_size_mb": file_size / 1024 / 1024,
            "file_hash": file_hash,
            "dataset_name": dataset_name,
            "processed_at": datetime.utcnow().isoformat(),
            "source_type": self._get_source_type(file_path.suffix.lower()),
            "large_file": True,
            **(metadata or {})
        }
        
        # Process content in chunks with optimal size
        chunks = self._chunk_text_streaming(content, base_metadata, chunk_size, chunk_overlap)
        
        batch = []
        total_chunks = 0
        
        for chunk in chunks:
            batch.append(chunk)
            total_chunks += 1
            
            if len(batch) >= self.max_chunks_per_batch:
                yield {
                    "chunks": batch,
                    "metadata": base_metadata,
                    "batch_size": len(batch),
                    "total_chunks_so_far": total_chunks
                }
                batch = []
        
        # Yield remaining chunks
        if batch:
            yield {
                "chunks": batch,
                "metadata": base_metadata,
                "batch_size": len(batch),
                "total_chunks_so_far": total_chunks
            }
    
    def _chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create chunks from text."""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        chunks = []
        text_length = len(text)
        
        if text_length <= chunk_size:
            chunks.append({
                "content": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "chunk_size": len(text)
                }
            })
        else:
            for i in range(0, text_length, chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                
                if len(chunk_text.strip()) < 10:
                    continue
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_size": len(chunk_text),
                        "chunk_start": i,
                        "chunk_end": min(i + chunk_size, text_length)
                    }
                })
        
        return chunks
    
    def _chunk_text_streaming(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate chunks from text without loading all in memory."""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        text_length = len(text)
        chunk_index = 0
        
        if text_length <= chunk_size:
            yield {
                "content": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "chunk_size": len(text)
                }
            }
        else:
            for i in range(0, text_length, chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                
                if len(chunk_text.strip()) < 10:
                    continue
                
                yield {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "chunk_start": i,
                        "chunk_end": min(i + chunk_size, text_length)
                    }
                }
                chunk_index += 1
    
    def _calculate_file_hash_streaming(
        self,
        file_path: Path,
        chunk_size: int = 8192
    ) -> str:
        """Calculate file hash using streaming to handle large files."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _get_source_type(self, extension: str) -> str:
        """Get source type from file extension."""
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'word',
            '.doc': 'word',
            '.pptx': 'powerpoint',
            '.ppt': 'powerpoint',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.csv': 'csv',
            '.txt': 'text',
            '.md': 'markdown',
            '.html': 'html',
            '.xml': 'xml',
            '.rtf': 'rtf',
            '.odt': 'opendocument',
            '.epub': 'epub'
        }
        return type_map.get(extension, 'unknown')
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if a file can be processed.
        
        Returns:
            Validation result with details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "valid": False,
                "error": "File does not exist"
            }
        
        file_size = file_path.stat().st_size
        
        if file_size > self.max_file_size:
            return {
                "valid": False,
                "error": f"File too large: {file_size / 1024 / 1024:.2f}MB > {self.max_file_size / 1024 / 1024:.2f}MB",
                "file_size_mb": file_size / 1024 / 1024,
                "max_size_mb": self.max_file_size / 1024 / 1024
            }
        
        # Check if it's a PDF and can be opened
        if file_path.suffix.lower() == '.pdf':
            try:
                with open(file_path, 'rb') as f:
                    pypdf.PdfReader(f)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Invalid or corrupted PDF: {str(e)}"
                }
        
        return {
            "valid": True,
            "file_size_mb": file_size / 1024 / 1024,
            "file_type": file_path.suffix.lower(),
            "is_large": file_size > 50 * 1024 * 1024,
            "recommended_processor": "large" if file_size > 50 * 1024 * 1024 else "standard"
        }