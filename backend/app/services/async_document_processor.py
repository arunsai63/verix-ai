import os
import hashlib
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from markitdown import MarkItDown
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import aiofiles

logger = logging.getLogger(__name__)


class AsyncDocumentProcessor:
    """Handles asynchronous document conversion and processing using MarkItDown with parallel processing."""
    
    def __init__(self, max_workers: int = 4):
        self.converter = MarkItDown()
        self.supported_extensions = [
            '.pdf', '.docx', '.pptx', '.html', '.txt', '.md', 
            '.csv', '.xlsx', '.xml', '.rtf', '.odt', '.epub'
        ]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue()
        self.chunk_cache = {}
    
    async def process_file_async(
        self,
        file_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single file asynchronously and convert it to markdown with metadata.
        
        Args:
            file_path: Path to the file to process
            dataset_name: Name of the dataset this file belongs to
            metadata: Additional metadata to attach to the document
            
        Returns:
            Dictionary containing processed content and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Convert document in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.converter.convert,
                str(file_path)
            )
            
            markdown_content = result.text_content
            
            # Calculate file hash asynchronously
            file_hash = await self._calculate_file_hash_async(file_path)
            
            processed_doc = {
                "content": markdown_content,
                "metadata": {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_extension": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size,
                    "file_hash": file_hash,
                    "dataset_name": dataset_name,
                    "processed_at": datetime.utcnow().isoformat(),
                    "source_type": self._get_source_type(file_path.suffix.lower()),
                    **(metadata or {})
                },
                "chunks": []
            }
            
            # Chunk document asynchronously
            processed_doc["chunks"] = await self._chunk_document_async(
                markdown_content,
                processed_doc["metadata"]
            )
            
            logger.info(f"Successfully processed file: {file_path.name}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    async def process_batch_async(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel batches.
        
        Args:
            file_paths: List of file paths to process
            dataset_name: Name of the dataset these files belong to
            metadata: Additional metadata to attach to all documents
            batch_size: Number of files to process concurrently
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        errors = []
        
        # Process files in batches to control memory usage
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = [
                self.process_file_async(file_path, dataset_name, metadata)
                for file_path in batch
            ]
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for file_path, result in zip(batch, results):
                if isinstance(result, Exception):
                    errors.append({
                        "file": file_path,
                        "error": str(result)
                    })
                    logger.error(f"Failed to process {file_path}: {str(result)}")
                else:
                    processed_docs.append(result)
        
        if errors:
            logger.warning(f"Batch processing completed with {len(errors)} errors")
        
        return processed_docs
    
    async def _chunk_document_async(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks asynchronously for embedding and retrieval.
        
        Args:
            content: The document content to chunk
            metadata: Metadata to attach to each chunk
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        # Use cache if available
        cache_key = hashlib.md5(f"{content[:100]}{chunk_size}{chunk_overlap}".encode()).hexdigest()
        if cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "start_char": sum(len(c["content"]) for c in chunks),
                    }
                })
                
                # Handle overlap
                if chunk_overlap > 0:
                    overlap_lines = []
                    overlap_size = 0
                    for line in reversed(current_chunk):
                        overlap_size += len(line)
                        if overlap_size >= chunk_overlap:
                            break
                        overlap_lines.insert(0, line)
                    current_chunk = overlap_lines
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
                
                chunk_index += 1
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                    "start_char": sum(len(c["content"]) for c in chunks),
                }
            })
        
        # Cache the result
        self.chunk_cache[cache_key] = chunks
        
        return chunks
    
    async def _calculate_file_hash_async(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file asynchronously for deduplication."""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                data = await f.read(8192)
                if not data:
                    break
                sha256_hash.update(data)
        
        return sha256_hash.hexdigest()
    
    def _get_source_type(self, extension: str) -> str:
        """Map file extension to source type category."""
        type_mapping = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.pptx': 'presentation',
            '.ppt': 'presentation',
            '.html': 'webpage',
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'spreadsheet',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.xml': 'structured_data',
            '.json': 'structured_data',
            '.rtf': 'document',
            '.odt': 'document',
            '.epub': 'ebook'
        }
        return type_mapping.get(extension, 'unknown')
    
    async def extract_citations_async(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract potential citations from document content asynchronously.
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of extracted citations with locations
        """
        citations = []
        lines = content.split('\n')
        
        # Process lines in parallel for large documents
        if len(lines) > 100:
            # Process in chunks for very large documents
            chunk_size = 50
            tasks = []
            
            for i in range(0, len(lines), chunk_size):
                chunk = lines[i:i + chunk_size]
                tasks.append(self._process_citation_chunk(chunk, i))
            
            results = await asyncio.gather(*tasks)
            for chunk_citations in results:
                citations.extend(chunk_citations)
        else:
            # Process small documents synchronously
            citations = await self._process_citation_chunk(lines, 0)
        
        return citations
    
    async def _process_citation_chunk(self, lines: List[str], start_line: int) -> List[Dict[str, Any]]:
        """Process a chunk of lines for citations."""
        citations = []
        
        for line_num, line in enumerate(lines, start_line + 1):
            if any(indicator in line.lower() for indicator in ['source:', 'ref:', 'citation:', 'reference:']):
                citations.append({
                    "line_number": line_num,
                    "text": line.strip(),
                    "type": "explicit_reference"
                })
            
            if line.strip().startswith('[') and ']' in line:
                ref_end = line.index(']')
                reference = line[1:ref_end]
                if reference and not reference.isspace():
                    citations.append({
                        "line_number": line_num,
                        "text": reference,
                        "type": "bracketed_reference"
                    })
        
        return citations
    
    async def process_with_progress(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process files with progress tracking.
        
        Args:
            file_paths: List of file paths to process
            dataset_name: Name of the dataset
            metadata: Additional metadata
            progress_callback: Callback function for progress updates
            
        Returns:
            List of processed documents
        """
        total_files = len(file_paths)
        processed_count = 0
        processed_docs = []
        
        for file_path in file_paths:
            try:
                doc = await self.process_file_async(file_path, dataset_name, metadata)
                processed_docs.append(doc)
                processed_count += 1
                
                if progress_callback:
                    await progress_callback({
                        "processed": processed_count,
                        "total": total_files,
                        "current_file": Path(file_path).name,
                        "percentage": (processed_count / total_files) * 100
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                
                if progress_callback:
                    await progress_callback({
                        "processed": processed_count,
                        "total": total_files,
                        "error": str(e),
                        "failed_file": Path(file_path).name
                    })
        
        return processed_docs
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self.chunk_cache.clear()