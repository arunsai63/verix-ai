"""
Hierarchical Chunking implementation for preserving document structure.
Creates parent-child relationships between chunks to maintain context.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalNode:
    """Node in the hierarchical document structure."""
    chunk_id: str
    content: str
    level: int
    title: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker:
    """
    Hierarchical chunking that preserves document structure.
    
    Creates a tree structure of chunks where each level represents
    a different granularity (document -> chapter -> section -> paragraph).
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        preserve_structure: bool = True,
        include_parent_context: bool = True
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            max_chunk_size: Maximum size for any chunk
            min_chunk_size: Minimum size for any chunk
            preserve_structure: Whether to preserve document structure
            include_parent_context: Include parent context in chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_structure = preserve_structure
        self.include_parent_context = include_parent_context
        
        # Markdown heading patterns
        self.heading_patterns = [
            (r'^# (.+)$', 1),           # Level 1: #
            (r'^## (.+)$', 2),          # Level 2: ##
            (r'^### (.+)$', 3),         # Level 3: ###
            (r'^#### (.+)$', 4),        # Level 4: ####
            (r'^##### (.+)$', 5),       # Level 5: #####
            (r'^###### (.+)$', 6),      # Level 6: ######
        ]
        
        logger.info("HierarchicalChunker initialized")
    
    def detect_structure(self, content: str) -> Dict[str, Any]:
        """
        Detect document structure from content.
        
        Args:
            content: Document content
            
        Returns:
            Document structure with chapters, sections, etc.
        """
        lines = content.split('\n')
        structure = {
            'type': self._detect_document_type(content),
            'chapters': [],
            'flat_sections': []
        }
        
        current_chapter = None
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for headings
            heading_level = self._get_heading_level(line)
            
            if heading_level == 1:
                # New chapter
                if current_chapter:
                    if current_section:
                        current_section['content'] = '\n'.join(current_content)
                        current_content = []
                        current_section = None
                    structure['chapters'].append(current_chapter)
                
                current_chapter = {
                    'title': self._extract_heading_text(line),
                    'level': 1,
                    'sections': []
                }
            
            elif heading_level == 2:
                # New section
                if current_section:
                    current_section['content'] = '\n'.join(current_content)
                    current_content = []
                
                current_section = {
                    'title': self._extract_heading_text(line),
                    'level': 2,
                    'content': ''
                }
                
                if current_chapter:
                    current_chapter['sections'].append(current_section)
                else:
                    structure['flat_sections'].append(current_section)
            
            elif heading_level > 2:
                # Subsection - add to current content
                current_content.append(line)
            
            else:
                # Regular content
                current_content.append(line)
        
        # Add final content
        if current_content:
            if current_section:
                current_section['content'] = '\n'.join(current_content)
            elif current_chapter:
                current_chapter['content'] = '\n'.join(current_content)
        
        if current_chapter:
            structure['chapters'].append(current_chapter)
        
        return structure
    
    def _detect_document_type(self, content: str) -> str:
        """Detect the type of document."""
        if re.search(r'^#+ ', content, re.MULTILINE):
            return 'markdown'
        elif re.search(r'<[^>]+>', content):
            return 'html'
        else:
            return 'text'
    
    def _get_heading_level(self, line: str) -> int:
        """Get heading level from a line."""
        for pattern, level in self.heading_patterns:
            if re.match(pattern, line.strip()):
                return level
        return 0
    
    def _extract_heading_text(self, line: str) -> str:
        """Extract text from a heading line."""
        # Remove markdown heading markers
        text = re.sub(r'^#+\s*', '', line.strip())
        return text
    
    def create_hierarchical_chunks(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        include_parent_context: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from document.
        
        Args:
            content: Document content
            metadata: Document metadata
            include_parent_context: Override default setting
            
        Returns:
            List of hierarchical chunks
        """
        include_parent_context = (
            include_parent_context 
            if include_parent_context is not None 
            else self.include_parent_context
        )
        
        # Detect structure
        structure = self.detect_structure(content)
        
        # Create nodes
        nodes = []
        root_id = str(uuid.uuid4())
        
        # Create root node
        root_node = HierarchicalNode(
            chunk_id=root_id,
            content=content[:self.max_chunk_size] if len(content) > self.max_chunk_size else content,
            level=0,
            title=metadata.get('title', 'Document') if metadata else 'Document',
            parent_id=None,
            metadata={**(metadata or {}), 'structure_type': structure['type']}
        )
        nodes.append(root_node)
        
        # Process chapters
        for chapter in structure['chapters']:
            chapter_node = self._create_chapter_node(
                chapter,
                root_id,
                metadata
            )
            nodes.append(chapter_node)
            root_node.children_ids.append(chapter_node.chunk_id)
            
            # Process sections
            for section in chapter.get('sections', []):
                section_node = self._create_section_node(
                    section,
                    chapter_node.chunk_id,
                    metadata
                )
                nodes.append(section_node)
                chapter_node.children_ids.append(section_node.chunk_id)
                
                # Split large sections into paragraphs
                if len(section.get('content', '')) > self.max_chunk_size:
                    paragraph_nodes = self._split_into_paragraphs(
                        section['content'],
                        section_node.chunk_id,
                        level=3,
                        metadata=metadata
                    )
                    nodes.extend(paragraph_nodes)
                    section_node.children_ids.extend([n.chunk_id for n in paragraph_nodes])
        
        # Process flat sections
        for section in structure['flat_sections']:
            section_node = self._create_section_node(
                section,
                root_id,
                metadata
            )
            nodes.append(section_node)
            root_node.children_ids.append(section_node.chunk_id)
        
        # Convert nodes to chunks
        chunks = self._nodes_to_chunks(nodes, include_parent_context)
        
        return chunks
    
    def _create_chapter_node(
        self,
        chapter: Dict[str, Any],
        parent_id: str,
        metadata: Optional[Dict[str, Any]]
    ) -> HierarchicalNode:
        """Create a chapter-level node."""
        content = chapter.get('content', '')
        if not content and chapter.get('sections'):
            # Use first section's content as preview
            content = chapter['sections'][0].get('content', '')[:self.max_chunk_size]
        
        return HierarchicalNode(
            chunk_id=str(uuid.uuid4()),
            content=content[:self.max_chunk_size],
            level=1,
            title=chapter.get('title', 'Chapter'),
            parent_id=parent_id,
            metadata={**(metadata or {}), 'chapter': chapter.get('title')}
        )
    
    def _create_section_node(
        self,
        section: Dict[str, Any],
        parent_id: str,
        metadata: Optional[Dict[str, Any]]
    ) -> HierarchicalNode:
        """Create a section-level node."""
        return HierarchicalNode(
            chunk_id=str(uuid.uuid4()),
            content=section.get('content', '')[:self.max_chunk_size],
            level=2,
            title=section.get('title', 'Section'),
            parent_id=parent_id,
            metadata={**(metadata or {}), 'section': section.get('title')}
        )
    
    def _split_into_paragraphs(
        self,
        content: str,
        parent_id: str,
        level: int,
        metadata: Optional[Dict[str, Any]]
    ) -> List[HierarchicalNode]:
        """Split large content into paragraph nodes."""
        paragraphs = content.split('\n\n')
        nodes = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < self.min_chunk_size:
                continue
            
            node = HierarchicalNode(
                chunk_id=str(uuid.uuid4()),
                content=paragraph[:self.max_chunk_size],
                level=level,
                title=f"Paragraph {i+1}",
                parent_id=parent_id,
                metadata={**(metadata or {}), 'paragraph_index': i}
            )
            nodes.append(node)
        
        return nodes
    
    def _nodes_to_chunks(
        self,
        nodes: List[HierarchicalNode],
        include_parent_context: bool
    ) -> List[Dict[str, Any]]:
        """Convert hierarchical nodes to chunks."""
        # Create lookup map
        node_map = {node.chunk_id: node for node in nodes}
        
        chunks = []
        for node in nodes:
            chunk = {
                'content': node.content,
                'metadata': {
                    **node.metadata,
                    'chunk_id': node.chunk_id,
                    'level': node.level,
                    'title': node.title,
                    'parent_id': node.parent_id,
                    'children_ids': node.children_ids,
                    'chunk_type': 'hierarchical'
                }
            }
            
            # Add parent context if requested
            if include_parent_context and node.parent_id:
                parent_context = []
                current_id = node.parent_id
                
                while current_id and current_id in node_map:
                    parent = node_map[current_id]
                    parent_context.append({
                        'level': parent.level,
                        'title': parent.title,
                        'preview': parent.content[:100] + '...' if len(parent.content) > 100 else parent.content
                    })
                    current_id = parent.parent_id
                
                chunk['metadata']['parent_context'] = list(reversed(parent_context))
            
            chunks.append(chunk)
        
        return chunks
    
    def recursive_split(
        self,
        content: str,
        max_size: int,
        level: int = 0,
        parent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively split content into hierarchical chunks.
        
        Args:
            content: Content to split
            max_size: Maximum chunk size
            level: Current hierarchy level
            parent_id: Parent chunk ID
            
        Returns:
            List of chunks
        """
        if len(content) <= max_size:
            return [{
                'content': content,
                'metadata': {
                    'chunk_id': str(uuid.uuid4()),
                    'level': level,
                    'parent_id': parent_id,
                    'chunk_type': 'recursive'
                }
            }]
        
        # Split content
        chunks = []
        parts = self._smart_split(content, max_size)
        
        parent_chunk_id = str(uuid.uuid4())
        
        # Create parent chunk (summary or first part)
        parent_chunk = {
            'content': parts[0][:max_size],
            'metadata': {
                'chunk_id': parent_chunk_id,
                'level': level,
                'parent_id': parent_id,
                'children_ids': [],
                'chunk_type': 'recursive_parent'
            }
        }
        chunks.append(parent_chunk)
        
        # Create child chunks
        for part in parts:
            if len(part) > max_size:
                # Recursively split
                sub_chunks = self.recursive_split(
                    part,
                    max_size,
                    level + 1,
                    parent_chunk_id
                )
                chunks.extend(sub_chunks)
                parent_chunk['metadata']['children_ids'].extend(
                    [c['metadata']['chunk_id'] for c in sub_chunks]
                )
            else:
                # Create leaf chunk
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'content': part,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'level': level + 1,
                        'parent_id': parent_chunk_id,
                        'chunk_type': 'recursive_leaf'
                    }
                })
                parent_chunk['metadata']['children_ids'].append(chunk_id)
        
        return chunks
    
    def _smart_split(self, content: str, max_size: int) -> List[str]:
        """Smart splitting that preserves sentence/paragraph boundaries."""
        # Try splitting by paragraphs first
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            return self._merge_to_size(paragraphs, max_size, '\n\n')
        
        # Try splitting by sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            return self._merge_to_size(sentences, max_size, ' ')
        
        # Fall back to hard split
        parts = []
        for i in range(0, len(content), max_size):
            parts.append(content[i:i+max_size])
        return parts
    
    def _merge_to_size(
        self,
        elements: List[str],
        max_size: int,
        separator: str
    ) -> List[str]:
        """Merge elements up to max size."""
        parts = []
        current = []
        current_size = 0
        
        for element in elements:
            element_size = len(element)
            
            if current_size + element_size + len(separator) > max_size and current:
                parts.append(separator.join(current))
                current = [element]
                current_size = element_size
            else:
                current.append(element)
                current_size += element_size + len(separator)
        
        if current:
            parts.append(separator.join(current))
        
        return parts
    
    def extract_hierarchy(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract hierarchical structure from chunks.
        
        Args:
            chunks: List of hierarchical chunks
            
        Returns:
            Tree structure representation
        """
        # Build tree structure
        tree = {}
        node_map = {}
        
        # First pass: create all nodes
        for chunk in chunks:
            chunk_id = chunk['metadata']['chunk_id']
            node_map[chunk_id] = {
                'id': chunk_id,
                'title': chunk['metadata'].get('title', 'Untitled'),
                'level': chunk['metadata'].get('level', 0),
                'children': []
            }
        
        # Second pass: build relationships
        root_nodes = []
        for chunk in chunks:
            chunk_id = chunk['metadata']['chunk_id']
            parent_id = chunk['metadata'].get('parent_id')
            
            if parent_id and parent_id in node_map:
                node_map[parent_id]['children'].append(node_map[chunk_id])
            else:
                root_nodes.append(node_map[chunk_id])
        
        return {
            'roots': root_nodes,
            'total_nodes': len(node_map),
            'max_depth': max(n['level'] for n in node_map.values()) if node_map else 0
        }