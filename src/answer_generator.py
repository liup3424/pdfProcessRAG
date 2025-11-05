"""
Answer Generation Module
Generates final answers based on retrieved documents using an LLM.
"""
from typing import List, Dict, Optional
import logging
import requests
import os
from . import config


class AnswerGenerator:
    """Generate answers from retrieved documents using LLM."""
    
    def __init__(self, llm_api_url: str = None):
        """
        Initialize answer generator.
        
        Args:
            llm_api_url: URL of the LLM API endpoint
        """
        self.llm_api_url = llm_api_url or config.LLM_API_URL
        self.llm_api_key = config.LLM_API_KEY
        self.model = config.LLM_MODEL
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_length: int = 2000
    ) -> str:
        """
        Generate answer based on query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document chunks
            max_context_length: Maximum characters of context to include
            
        Returns:
            Generated answer text
        """
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs, max_context_length)
        
        # Generate answer using LLM (only if API URL and API key are provided)
        if self.llm_api_url and self.llm_api_key:
            return self._generate_with_api(query, context)
        else:
            if self.llm_api_url and not self.llm_api_key:
                logging.getLogger(__name__).warning("LLM_API_URL is set but LLM_API_KEY is missing. Using simple answer generator.")
            return self._generate_simple_answer(query, context, retrieved_docs)
    
    def _build_context(self, docs: List[Dict], max_length: int) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            docs: List of document dictionaries
            max_length: Maximum context length
            
        Returns:
            Context string
        """
        context_parts = []
        current_length = 0
        
        for doc in docs:
            doc_text = doc.get("text", "")
            doc_length = len(doc_text)
            
            if current_length + doc_length > max_length:
                # Add partial document if there's room
                remaining = max_length - current_length
                if remaining > 100:  # Only add if there's meaningful space
                    context_parts.append(doc_text[:remaining])
                break
            
            context_parts.append(doc_text)
            current_length += doc_length
        
        return "\n\n".join(context_parts)
    
    def _generate_with_api(self, query: str, context: str) -> str:
        """
        Generate answer using LLM API.
        
        Args:
            query: User query
            context: Context from retrieved documents
            
        Returns:
            Generated answer
        """
        try:
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            # Call LLM API
            # Determine if model uses max_tokens or max_completion_tokens
            use_max_completion_tokens = (
                "gpt-4o" in self.model.lower() or 
                "gpt-5" in self.model.lower() or
                "o1" in self.model.lower()
            )
            
            # Some models only support default temperature (1)
            # Models like o1, gpt-4o-mini, and some newer models have temperature restrictions
            model_lower = self.model.lower()
            supports_custom_temperature = not (
                "o1" in model_lower or
                "gpt-4o-mini" in model_lower or
                ("gpt-4o" in model_lower and "2024" in self.model)
            )
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information, say so."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Add temperature only if model supports it
            if supports_custom_temperature:
                payload["temperature"] = 0.7
            else:
                # Use default temperature (1) for models that don't support custom values
                payload["temperature"] = 1
            
            # Add token limit based on model type
            # For models that use reasoning tokens (like gpt-5-nano), we need higher limits
            # since reasoning tokens count towards the limit
            if use_max_completion_tokens:
                # Increase limit to account for reasoning tokens
                # Reasoning tokens can take 200-500 tokens, so we need at least 1000+ for content
                payload["max_completion_tokens"] = 2000
            else:
                payload["max_tokens"] = 500
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key authentication if provided
            if self.llm_api_key:
                # OpenAI uses "Authorization: Bearer <key>"
                if "openai.com" in self.llm_api_url or "api.openai" in self.llm_api_url:
                    headers["Authorization"] = f"Bearer {self.llm_api_key}"
                # Anthropic uses "x-api-key" header
                elif "anthropic.com" in self.llm_api_url:
                    headers["x-api-key"] = self.llm_api_key
                    headers["anthropic-version"] = "2023-06-01"
                # Generic API key header
                else:
                    headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            # Try with initial payload
            response = requests.post(
                self.llm_api_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            
            # If we get a temperature error, retry with temperature=1
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", "")
                        error_param = error_data["error"].get("param", "")
                        # Check if it's a temperature error
                        if "temperature" in error_msg.lower() or error_param == "temperature":
                            # Retry with temperature=1 (default)
                            payload["temperature"] = 1
                            response = requests.post(
                                self.llm_api_url,
                                json=payload,
                                headers=headers,
                                timeout=60
                            )
                except (ValueError, KeyError):
                    pass  # If we can't parse the error, continue with original response
            
            # Log error details if request fails
            if response.status_code != 200:
                logger = logging.getLogger(__name__)
                logger.error("LLM API Error: %s", response.status_code)
                logger.error("Response: %s", response.text[:500])
            
            response.raise_for_status()
            
            result = response.json()
            
            # Debug: Log the response structure
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"LLM API Response keys: {list(result.keys())}")
            if "choices" in result and result["choices"]:
                logger.debug(f"First choice keys: {list(result['choices'][0].keys())}")
            
            # Handle different response formats
            if "choices" in result and result["choices"]:
                message = result["choices"][0].get("message", {})
                logger.debug(f"Message type: {type(message)}, keys: {list(message.keys()) if isinstance(message, dict) else 'N/A'}")
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    logger.debug(f"Content length: {len(content) if content else 0}")
                    return content or ""
                elif isinstance(message, str):
                    return message
                else:
                    logger.warning(f"Message format unexpected: {message}")
            elif "content" in result:
                return result["content"]
            elif "text" in result:
                return result["text"]
            else:
                logger.warning(f"Unexpected API response format. Keys: {list(result.keys())}")
                return "Error: Unexpected API response format"
                
        except requests.exceptions.RequestException as e:
            logging.getLogger(__name__).error("Error calling LLM API: %s", e, exc_info=True)
            return self._generate_simple_answer(query, context, [])
        except Exception as e:
            logging.getLogger(__name__).error("Error generating answer: %s", e, exc_info=True)
            return self._generate_simple_answer(query, context, [])
    
    def _generate_simple_answer(
        self,
        query: str,
        context: str,
        docs: List[Dict]
    ) -> str:
        """
        Generate a simple answer without LLM API (template-based).
        
        Args:
            query: User query
            context: Context from retrieved documents
            docs: Retrieved documents
            
        Returns:
            Simple answer
        """
        if not context:
            return "I couldn't find any relevant information to answer your question."
        
        # Simple template-based answer
        answer = f"Based on the retrieved documents, here is relevant information:\n\n"
        answer += context[:500]  # Limit to first 500 characters
        
        if len(docs) > 0:
            answer += f"\n\n[Retrieved from {len(docs)} document(s)]"
        
        answer += "\n\nNote: For better answers, please configure an LLM API in the config file."
        
        return answer
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            query: User query
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate_answer_with_sources(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_length: int = 2000
    ) -> Dict:
        """
        Generate answer with source citations.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document chunks
            max_context_length: Maximum context length
            
        Returns:
            Dictionary with answer and sources
        """
        answer = self.generate_answer(query, retrieved_docs, max_context_length)
        
        # Extract source information
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            source = {
                "text": doc.get("text", "")[:200],  # First 200 chars
                "file_name": metadata.get("file_name", "Unknown"),
                "page_number": metadata.get("page_number", None),
                "chunk_id": doc.get("chunk_id", None)
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }

