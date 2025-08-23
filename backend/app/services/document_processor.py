import os
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from markitdown import MarkItDown
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document conversion and processing using MarkItDown."""
    
    def __init__(self):
        self.converter = MarkItDown()
        self.supported_extensions = [
            '.pdf', '.docx', '.pptx', '.html', '.txt', '.md', 
            '.csv', '.xlsx', '.xml', '.rtf', '.odt', '.epub'
        ]
    
    def process_file(
        self,
        file_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single file and convert it to markdown with metadata.
        
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
            
            result = self.converter.convert(str(file_path))
            
            markdown_content = result.text_content
            
            file_hash = self._calculate_file_hash(file_path)
            
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
            
            processed_doc["chunks"] = self._chunk_document(
                markdown_content,
                processed_doc["metadata"]
            )
            
            logger.info(f"Successfully processed file: {file_path.name}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def process_batch(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            dataset_name: Name of the dataset these files belong to
            metadata: Additional metadata to attach to all documents
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        errors = []
        
        for file_path in file_paths:
            try:
                doc = self.process_file(file_path, dataset_name, metadata)
                processed_docs.append(doc)
            except Exception as e:
                errors.append({
                    "file": file_path,
                    "error": str(e)
                })
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        if errors:
            logger.warning(f"Batch processing completed with {len(errors)} errors")
        
        return processed_docs
    
    def _chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks for embedding and retrieval.
        
        Args:
            content: The document content to chunk
            metadata: Metadata to attach to each chunk
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
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
        
        return chunks
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
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
    
    def extract_citations(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract potential citations from document content.
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of extracted citations with locations
        """
        citations = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
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