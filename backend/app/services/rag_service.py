import logging
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from app.core.config import settings
from app.services.ai_providers import AIProviderFactory
import json

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service for answer generation."""
    
    def __init__(self):
        provider = AIProviderFactory.get_provider()
        self.llm = provider.get_chat_model(
            temperature=0.3,
            max_tokens=2000
        )
        
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
                "system": """You are an AI assistant helping analyze documents and provide accurate, well-cited answers.
                Be clear, concise, and always provide sources for your information.""",
                "tone": "clear, helpful, informative"
            }
        }
    
    async def generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        role: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved documents.
        
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
            
            context = self._prepare_context(search_results)
            
            role_config = self.role_prompts.get(role, self.role_prompts["general"])
            
            prompt = self._create_prompt(role_config["system"])
            
            response = await self._generate_response(
                prompt, query, context, role_config["tone"]
            )
            
            parsed_response = self._parse_response(response, search_results)
            
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
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results."""
        context_parts = []
        
        for idx, result in enumerate(search_results[:5], 1):
            source = result["metadata"].get("filename", "Unknown")
            chunk_index = result["metadata"].get("chunk_index", 0)
            content = result.get("full_content", result["content"])
            
            context_parts.append(
                f"[Source {idx}] {source} (Chunk {chunk_index}):\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, system_message: str) -> ChatPromptTemplate:
        """Create the prompt template."""
        system_template = SystemMessagePromptTemplate.from_template(system_message)
        
        human_template = """Based on the following context, please answer the question. 
        Always cite your sources using [Source N] format where N is the source number.
        If you're not confident about something, say so.
        
        Context:
        {context}
        
        Question: {query}
        
        Please format your response as JSON with the following structure:
        {{
            "answer": "Your detailed answer with [Source N] citations",
            "highlights": ["Key point 1 with citation", "Key point 2 with citation"],
            "confidence": "high/medium/low",
            "suggested_followup": "Optional suggested follow-up question"
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
        
        response = await chain.arun(
            query=query,
            context=context,
            tone=tone
        )
        
        return response
    
    def _parse_response(
        self,
        response: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response and extract citations."""
        try:
            parsed = json.loads(response)
            
            citations = []
            for idx, result in enumerate(search_results[:5], 1):
                if f"[Source {idx}]" in parsed.get("answer", ""):
                    citations.append({
                        "source_number": idx,
                        "filename": result["metadata"].get("filename"),
                        "dataset": result["metadata"].get("dataset_name"),
                        "chunk_index": result["metadata"].get("chunk_index"),
                        "snippet": result["content"][:200] + "...",
                        "relevance_score": result.get("score", 0)
                    })
            
            return {
                "answer": parsed.get("answer", "Unable to generate answer"),
                "citations": citations,
                "highlights": parsed.get("highlights", []),
                "confidence": parsed.get("confidence", "medium"),
                "suggested_followup": parsed.get("suggested_followup")
            }
            
        except json.JSONDecodeError:
            return {
                "answer": response,
                "citations": [],
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