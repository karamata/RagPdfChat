import requests
import json
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralClient:
    """
    Client for interacting with Mistral API
    """
    
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        """
        Initialize Mistral client
        
        Args:
            api_key: Mistral API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        
        # Set up headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], max_tokens: int = 1000) -> str:
        """
        Generate response using Mistral API with RAG context
        
        Args:
            query: User query
            context_chunks: Relevant context chunks from vector search
            max_tokens: Maximum tokens in response
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare context from chunks
            context = self._prepare_context(context_chunks)
            
            # Create system message with context
            system_message = f"""You are a helpful AI assistant that answers questions based on the provided document context. 

Context from uploaded documents:
{context}

Instructions:
- Answer the user's question using only the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but thorough in your response
- Include relevant details from the context when applicable
- If you reference specific information, mention which document it came from if available"""

            # Prepare messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # Make API request
            logger.info(f"Making request to Mistral API with model: {self.model}")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"Mistral API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"I encountered an error while generating a response: {error_msg}"
            
            # Parse response
            response_data = response.json()
            
            if 'choices' not in response_data or not response_data['choices']:
                logger.error("Invalid response format from Mistral API")
                return "I received an invalid response format from the AI service. Please try again."
            
            generated_text = response_data['choices'][0]['message']['content'].strip()
            
            logger.info("Successfully generated response using Mistral API")
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("Timeout error when calling Mistral API")
            return "The request timed out. Please try again with a shorter question."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when calling Mistral API: {str(e)}")
            return f"I encountered a network error: {str(e)}. Please check your connection and try again."
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return "I received an invalid response from the AI service. Please try again."
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return f"An unexpected error occurred: {str(e)}. Please try again."
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """
        Prepare context string from retrieved chunks
        
        Args:
            context_chunks: List of context chunks with metadata
            max_context_length: Maximum length of context string
            
        Returns:
            str: Formatted context string
        """
        if not context_chunks:
            return "No relevant context found in the uploaded documents."
        
        context_parts = []
        current_length = 0
        
        # Sort chunks by relevance score (descending)
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        for i, chunk in enumerate(sorted_chunks):
            # Format chunk with source information
            chunk_text = chunk.get('text', '').strip()
            filename = chunk.get('filename', 'Unknown')
            
            chunk_formatted = f"[From {filename}]\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_formatted) > max_context_length:
                break
            
            context_parts.append(chunk_formatted)
            current_length += len(chunk_formatted)
        
        if not context_parts:
            return "No relevant context could be included due to length constraints."
        
        context = "\n---\n".join(context_parts)
        
        logger.info(f"Prepared context with {len(context_parts)} chunks ({current_length} characters)")
        return context
    
    def test_api_connection(self) -> bool:
        """
        Test the connection to Mistral API
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Simple test request
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Mistral API connection test successful")
                return True
            else:
                logger.error(f"Mistral API connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Mistral API connection test error: {str(e)}")
            return False
