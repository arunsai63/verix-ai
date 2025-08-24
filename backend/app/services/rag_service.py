import logging
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from app.core.config import settings
from app.services.ai_providers import AIProviderFactory
import json
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service for answer generation."""
    
    def __init__(self, cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        provider = AIProviderFactory.get_provider()
        self.llm = provider.get_chat_model(
            temperature=0.1, # Lower temperature for more factual answers
            max_tokens=4096 # Increase max tokens for more comprehensive answers
        )
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        self.role_prompts = {
            "doctor": {
                "system": """You are an AI assistant helping medical professionals analyze patient documents and medical literature. 
                Always include appropriate medical disclaimers. Be precise with medical terminology.
                IMPORTANT: This is for informational purposes only and not a substitute for professional medical advice.""",
                "tone": "professional, clinical, precise"
            },
            "lawyer": {
                "system": """You are an AI assistant helping legal professionals analyze case files and legal documents.
                Always include appropriate legal disclaimers. Use precise legal terminology.
                IMPORTANT: This is for informational purposes only and not legal advice.""",
                "tone": "formal, precise, analytical"
            },
            "hr": {
                "system": """You are an AI assistant helping HR professionals analyze policies, employee documents, and compliance materials.
                Focus on actionable insights and compliance considerations.
                IMPORTANT: This is for informational purposes only. Consult with legal counsel for specific situations.""",
                "tone": "professional, clear, actionable"
            },
            "general": {
                "system": """You are a highly advanced AI assistant. Your goal is to provide the most accurate and comprehensive answer possible based on the provided context. 
                Analyze the context thoroughly, synthesize the information, and present it in a clear, well-structured manner. 
                Always cite your sources using [Source N] format where N is the source number.""",
                "tone": "analytical, comprehensive, precise"
            }
        }

    async def _expand_query(self, query: str) -> str:
        """Expand the user's query with related terms and concepts."""
        prompt = """You are a query expansion expert. Your task is to expand the following user query to improve retrieval accuracy. 
        Generate a new query that is a more detailed and comprehensive version of the original. 
        Include synonyms, related concepts, and rephrase the query to be more specific. 
        Original query: {query}
        Expanded query:"""
        
        expanded_query = await self.llm.apredict(prompt.format(query=query))
        return expanded_query.strip()

    def _rerank_results(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results using a CrossEncoder model."""
        if not search_results:
            return []

        pairs = [[query, result['content']] for result in search_results]
        scores = self.cross_encoder.predict(pairs)

        for result, score in zip(search_results, scores):
            result['rerank_score'] = score

        return sorted(search_results, key=lambda x: x['rerank_score'], reverse=True)

    async def generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        role: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved and reranked documents.
        
        Args:
            query: User's question
            search_results: Retrieved document chunks
            role: User role for response formatting
            
        Returns:
            Generated answer with citations
        """
        try:
            if not search_results:
                return {
                    "answer": "No relevant documents found to answer your query.",
                    "citations": [],
                    "highlights": [],
                    "confidence": "low"
                }

            # 1. Expand the query
            expanded_query = await self._expand_query(query)

            # 2. Rerank the search results
            reranked_results = self._rerank_results(expanded_query, search_results)
            
            # 3. Prepare context from top N reranked results
            context = self._prepare_context(reranked_results)
            
            role_config = self.role_prompts.get(role, self.role_prompts["general"])
            
            prompt = self._create_prompt(role_config["system"])
            
            response = await self._generate_response(
                prompt, query, context, role_config["tone"]
            )
            
            parsed_response = self._parse_response(response, reranked_results)
            
            if role != "general":
                parsed_response["disclaimer"] = self._get_disclaimer(role)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "An error occurred while generating the answer.",
                "error": str(e),
                "citations": [],
                "highlights": [],
                "confidence": "error"
            }
    
    def _prepare_context(self, search_results: List[Dict[str, Any]], max_context_size: int = 8000) -> str:
        """Prepare context from search results, prioritizing higher-ranked results."""
        context_parts = []
        total_size = 0
        
        for idx, result in enumerate(search_results, 1):
            source = result["metadata"].get("filename", "Unknown")
            chunk_index = result["metadata"].get("chunk_index", 0)
            content = result.get("full_content", result["content"])
            
            result_text = f"[Source {idx}] {source} (Chunk {chunk_index}):\n{content}\n"
            
            if total_size + len(result_text) > max_context_size:
                break

            context_parts.append(result_text)
            total_size += len(result_text)
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, system_message: str) -> ChatPromptTemplate:
        """Create the prompt template."""
        system_template = SystemMessagePromptTemplate.from_template(system_message)
        
        human_template = """Based on the following context, please provide a comprehensive and accurate answer to the question. 
        Synthesize information from multiple sources. If the context does not contain the answer, state that clearly.
        Always cite your sources using [Source N] format where N is the source number.
        
        Context:
        {context}
        
        Question: {query}
        
        Please format your response as JSON with the following structure:
        {{
            "answer": "Your detailed answer with [Source N] citations. Synthesize information from multiple sources to provide a comprehensive response.",
            "highlights": ["Key insight 1 with citation [Source N]", "Key insight 2 with citation [Source N]"],
            "confidence": "high/medium/low",
            "suggested_followup": "A relevant follow-up question based on the context."
        }}
        
        Tone: {tone}
        """
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_template, human_message])
    
    async def _generate_response(
        self,
        prompt: ChatPromptTemplate,
        query: str,
        context: str,
        tone: str
    ) -> str:
        """Generate response using LLM."""
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.ainvoke({
            "query": query,
            "context": context,
            "tone": tone
        })
        
        # Handle different response types properly
        if isinstance(response, dict):
            return response.get('text', str(response))
        elif isinstance(response, (list, tuple)):
            # If it's a tuple or list, convert to string
            return str(response[0]) if response else ""
        else:
            return str(response)
    
    def _parse_response(
        self,
        response: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response and extract citations."""
        try:
            parsed = json.loads(response)
            
            citations = []
            # Use the reranked search results for citations
            for idx, result in enumerate(search_results, 1):
                is_mentioned = f"[Source {idx}]" in parsed.get("answer", "")
                
                citations.append({
                    "source_number": idx,
                    "filename": result["metadata"].get("filename", "Unknown"),
                    "dataset": result["metadata"].get("dataset_name", "Unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "snippet": result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"],
                    "relevance_score": result.get("rerank_score", result.get("score", 0)),
                    "mentioned_in_answer": is_mentioned
                })
            
            return {
                "answer": parsed.get("answer", "Unable to generate answer"),
                "citations": citations,
                "highlights": parsed.get("highlights", []),
                "confidence": parsed.get("confidence", "medium"),
                "suggested_followup": parsed.get("suggested_followup")
            }
            
        except json.JSONDecodeError:
            citations = []
            for idx, result in enumerate(search_results, 1):
                citations.append({
                    "source_number": idx,
                    "filename": result["metadata"].get("filename", "Unknown"),
                    "dataset": result["metadata"].get("dataset_name", "Unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "snippet": result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"],
                    "relevance_score": result.get("rerank_score", result.get("score", 0)),
                    "mentioned_in_answer": False
                })
            
            return {
                "answer": response,
                "citations": citations,
                "highlights": [],
                "confidence": "low"
            }
    
    def _get_disclaimer(self, role: str) -> str:
        """Get role-specific disclaimer."""
        disclaimers = {
            "doctor": "⚠️ Medical Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.",
            "lawyer": "⚠️ Legal Disclaimer: This information is for educational purposes only and does not constitute legal advice. For specific legal advice, please consult with a qualified attorney who can consider the particular circumstances of your case.",
            "hr": "⚠️ HR Disclaimer: This information is for general guidance only and should not be considered as specific HR or legal advice. Please consult with qualified HR professionals or legal counsel for decisions regarding specific situations."
        }
        return disclaimers.get(role, "")
    
    async def generate_summary(
        self,
        documents: List[Dict[str, Any]],
        summary_type: str = "brief"
    ) -> str:
        """
        Generate a summary of multiple documents.
        
        Args:
            documents: List of documents to summarize
            summary_type: Type of summary (brief, detailed, executive)
            
        Returns:
            Generated summary
        """
        try:
            if summary_type == "brief":
                max_length = 200
                instruction = "Provide a brief 2-3 sentence summary"
            elif summary_type == "detailed":
                max_length = 500
                instruction = "Provide a detailed summary with key points"
            else:
                max_length = 300
                instruction = "Provide an executive summary with main findings"
            
            combined_content = "\n\n".join([
                doc.get("content", "")[:1000] for doc in documents[:5]
            ])
            
            prompt = f"{instruction} of the following content:\n\n{combined_content}"
            
            response = await self.llm.apredict(prompt, max_tokens=max_length)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary"
