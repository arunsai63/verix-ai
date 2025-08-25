"""
Document Summarization Service

This module provides comprehensive document summarization capabilities including:
- Multiple summary types (executive, key points, chapter-wise, technical)
- Configurable summary lengths
- Multi-format support
- Caching for performance
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from ..services.vector_store import VectorStoreService
from ..services.ai_providers import get_llm_provider
from ..core.config import settings

logger = logging.getLogger(__name__)


class SummaryType(str, Enum):
    """Types of summaries available"""
    EXECUTIVE = "executive"
    KEY_POINTS = "key_points"
    CHAPTER_WISE = "chapter_wise"
    TECHNICAL = "technical"
    BULLET_POINTS = "bullet_points"
    ABSTRACT = "abstract"


class SummaryLength(str, Enum):
    """Available summary lengths"""
    BRIEF = "brief"  # 1-2 paragraphs
    STANDARD = "standard"  # 1 page
    DETAILED = "detailed"  # 2-3 pages


@dataclass
class SummaryRequest:
    """Summary request configuration"""
    content: str
    summary_type: SummaryType
    length: SummaryLength
    dataset_name: Optional[str] = None
    document_name: Optional[str] = None
    custom_instructions: Optional[str] = None
    include_citations: bool = True
    language: str = "english"


@dataclass
class SummaryResponse:
    """Summary response with metadata"""
    summary: str
    summary_type: str
    length: str
    word_count: int
    key_topics: List[str]
    confidence_score: float
    processing_time: float
    cached: bool = False
    citations: Optional[List[Dict[str, Any]]] = None


class SummarizationService:
    """Advanced document summarization service"""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm_provider = get_llm_provider()
        self.cache = {}  # Simple in-memory cache
        self.summary_prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize prompts for different summary types"""
        return {
            SummaryType.EXECUTIVE: """
                Create an executive summary of the following document. Focus on:
                1. Main objectives and goals
                2. Key findings and insights
                3. Strategic recommendations
                4. Business impact and outcomes
                Keep it concise and actionable for senior leadership.
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """,
            
            SummaryType.KEY_POINTS: """
                Extract and summarize the key points from this document:
                1. List the most important points (5-10 depending on length)
                2. Provide brief explanation for each point
                3. Highlight critical information
                4. Note any action items or decisions
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """,
            
            SummaryType.CHAPTER_WISE: """
                Create a chapter-by-chapter or section-by-section summary:
                1. Identify main sections/chapters
                2. Summarize each section separately
                3. Maintain document structure
                4. Include section headings
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """,
            
            SummaryType.TECHNICAL: """
                Provide a technical summary focusing on:
                1. Technical specifications and details
                2. Methodologies and approaches
                3. Data and metrics
                4. Technical challenges and solutions
                5. Implementation details
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """,
            
            SummaryType.BULLET_POINTS: """
                Summarize the document in clear bullet points:
                - Use concise, informative bullets
                - Group related points together
                - Highlight key facts and figures
                - Include important dates and names
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """,
            
            SummaryType.ABSTRACT: """
                Write an academic-style abstract that includes:
                1. Background/Context
                2. Objectives/Purpose
                3. Methods/Approach
                4. Results/Findings
                5. Conclusions/Implications
                
                Document content:
                {content}
                
                Length: {length_instruction}
                Additional instructions: {custom_instructions}
            """
        }
    
    def _get_length_instruction(self, length: SummaryLength) -> str:
        """Get length-specific instructions"""
        instructions = {
            SummaryLength.BRIEF: "Keep the summary very brief, 1-2 paragraphs maximum (150-300 words).",
            SummaryLength.STANDARD: "Provide a standard length summary, approximately 1 page (500-750 words).",
            SummaryLength.DETAILED: "Create a comprehensive summary, 2-3 pages (1500-2000 words)."
        }
        return instructions.get(length, instructions[SummaryLength.STANDARD])
    
    def _generate_cache_key(self, request: SummaryRequest) -> str:
        """Generate cache key for summary request"""
        key_data = {
            "content_hash": hashlib.md5(request.content.encode()).hexdigest(),
            "summary_type": request.summary_type,
            "length": request.length,
            "custom_instructions": request.custom_instructions
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def summarize_document(self, request: SummaryRequest) -> SummaryResponse:
        """Generate document summary based on request parameters"""
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._generate_cache_key(request)
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            cached_response.cached = True
            logger.info(f"Returning cached summary for {request.summary_type}")
            return cached_response
        
        try:
            # Prepare prompt
            prompt = self.summary_prompts[request.summary_type].format(
                content=request.content[:50000],  # Limit content length
                length_instruction=self._get_length_instruction(request.length),
                custom_instructions=request.custom_instructions or "None"
            )
            
            # Generate summary using LLM
            summary = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=self._get_max_tokens(request.length)
            )
            
            # Extract key topics
            key_topics = await self._extract_key_topics(request.content)
            
            # Get citations if requested
            citations = None
            if request.include_citations and request.dataset_name:
                citations = await self._extract_citations(
                    summary, 
                    request.dataset_name,
                    request.document_name
                )
            
            # Calculate metrics
            word_count = len(summary.split())
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_confidence(summary, request.content)
            
            # Create response
            response = SummaryResponse(
                summary=summary,
                summary_type=request.summary_type,
                length=request.length,
                word_count=word_count,
                key_topics=key_topics,
                confidence_score=confidence_score,
                processing_time=processing_time,
                cached=False,
                citations=citations
            )
            
            # Cache the response
            self.cache[cache_key] = response
            
            logger.info(f"Generated {request.summary_type} summary in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    async def summarize_dataset(
        self, 
        dataset_name: str, 
        summary_type: SummaryType = SummaryType.EXECUTIVE,
        length: SummaryLength = SummaryLength.STANDARD
    ) -> SummaryResponse:
        """Summarize entire dataset of documents"""
        try:
            # Retrieve all documents from dataset
            documents = await self.vector_store.get_dataset_documents(dataset_name)
            
            if not documents:
                raise ValueError(f"No documents found in dataset: {dataset_name}")
            
            # Combine document contents
            combined_content = "\n\n".join([doc.get("content", "") for doc in documents])
            
            # Create summary request
            request = SummaryRequest(
                content=combined_content,
                summary_type=summary_type,
                length=length,
                dataset_name=dataset_name,
                custom_instructions=f"This is a summary of {len(documents)} documents from the {dataset_name} dataset."
            )
            
            return await self.summarize_document(request)
            
        except Exception as e:
            logger.error(f"Error summarizing dataset {dataset_name}: {str(e)}")
            raise
    
    async def generate_comparative_summary(
        self,
        documents: List[Dict[str, Any]],
        comparison_aspects: Optional[List[str]] = None
    ) -> SummaryResponse:
        """Generate comparative summary of multiple documents"""
        try:
            if not comparison_aspects:
                comparison_aspects = ["similarities", "differences", "key themes", "conclusions"]
            
            prompt = f"""
            Compare and contrast the following {len(documents)} documents:
            
            Documents:
            {json.dumps(documents, indent=2)}
            
            Focus on these aspects:
            {', '.join(comparison_aspects)}
            
            Provide a structured comparison highlighting:
            1. Common themes and agreements
            2. Contrasting viewpoints or differences
            3. Unique contributions from each document
            4. Overall synthesis and conclusions
            """
            
            summary = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=2000
            )
            
            return SummaryResponse(
                summary=summary,
                summary_type="comparative",
                length=SummaryLength.DETAILED,
                word_count=len(summary.split()),
                key_topics=comparison_aspects,
                confidence_score=0.85,
                processing_time=0,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Error generating comparative summary: {str(e)}")
            raise
    
    async def _extract_key_topics(self, content: str, max_topics: int = 10) -> List[str]:
        """Extract key topics from document content"""
        try:
            prompt = f"""
            Extract the {max_topics} most important topics from this document.
            Return only the topic names as a comma-separated list.
            
            Document: {content[:5000]}
            
            Topics:
            """
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=200
            )
            
            topics = [topic.strip() for topic in response.split(',')]
            return topics[:max_topics]
            
        except Exception as e:
            logger.error(f"Error extracting key topics: {str(e)}")
            return []
    
    async def _extract_citations(
        self, 
        summary: str, 
        dataset_name: str,
        document_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract citations for summary content"""
        try:
            # Search for relevant chunks that support the summary
            search_results_tuples = self.vector_store.search(
                query=summary[:1000],  # Use part of summary as query
                dataset_names=[dataset_name],
                k=5
            )
            
            citations = []
            for doc, score in search_results_tuples:
                citation = {
                    "source": doc.metadata.get("source", document_name),
                    "page": doc.metadata.get("page"),
                    "relevance_score": score,
                    "excerpt": doc.page_content[:200]
                }
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error extracting citations: {str(e)}")
            return []
    
    def _calculate_confidence(self, summary: str, original_content: str) -> float:
        """Calculate confidence score for generated summary"""
        # Simple heuristic based on summary length and content coverage
        summary_words = set(summary.lower().split())
        content_words = set(original_content.lower().split()[:1000])  # Check first 1000 words
        
        # Calculate word overlap
        overlap = len(summary_words.intersection(content_words))
        coverage = overlap / len(summary_words) if summary_words else 0
        
        # Adjust based on summary length
        length_factor = min(len(summary.split()) / 100, 1.0)  # Normalize to 0-1
        
        confidence = (coverage * 0.7) + (length_factor * 0.3)
        return min(max(confidence, 0.5), 0.95)  # Clamp between 0.5 and 0.95
    
    def _get_max_tokens(self, length: SummaryLength) -> int:
        """Get max tokens based on summary length"""
        tokens = {
            SummaryLength.BRIEF: 500,
            SummaryLength.STANDARD: 1000,
            SummaryLength.DETAILED: 3000
        }
        return tokens.get(length, 1000)
    
    def clear_cache(self):
        """Clear summary cache"""
        self.cache.clear()
        logger.info("Summary cache cleared")
    
    async def get_summary_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics about summaries for a dataset"""
        # This would typically query a database
        # For now, return mock statistics
        return {
            "dataset": dataset_name,
            "total_summaries": len([k for k in self.cache.keys()]),
            "summary_types": list(SummaryType),
            "average_processing_time": 2.5,
            "cache_hit_rate": 0.3
        }