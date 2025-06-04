import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from pdf_processor import PDFProcessor
from pinecone_manager import PineconeManager
from mistral_client import MistralClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system that combines PDF processing,
    vector storage, and LLM generation
    """
    
    def __init__(self, mistral_api_key: str, pinecone_api_key: str, index_name: str):
        """
        Initialize the RAG system
        
        Args:
            mistral_api_key: API key for Mistral
            pinecone_api_key: API key for Pinecone
            index_name: Name for the Pinecone index
        """
        self.pdf_processor = PDFProcessor()
        self.pinecone_manager = PineconeManager(
            api_key=pinecone_api_key,
            index_name=index_name
        )
        self.mistral_client = MistralClient(api_key=mistral_api_key)
        
        # Initialize Pinecone index
        self.pinecone_manager.initialize_index()
        
        logger.info("RAG system initialized successfully")
    
    def add_document(self, file_content: bytes, filename: str) -> bool:
        """
        Add a document to the RAG system
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the file
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text from document
            text = self.pdf_processor.extract_text_from_document(file_content, filename)
            if not text.strip():
                logger.error(f"No text extracted from {filename}")
                return False
            
            # Split text into chunks
            chunks = self.pdf_processor.split_text_into_chunks(text)
            logger.info(f"Split {filename} into {len(chunks)} chunks")
            
            # Generate embeddings and store in Pinecone
            success = self.pinecone_manager.add_chunks(chunks, filename)
            
            if success:
                logger.info(f"Successfully added {filename} to vector database")
                return True
            else:
                logger.error(f"Failed to add {filename} to vector database")
                return False
                
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> str:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            str: Generated response
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant chunks
            relevant_chunks = self.pinecone_manager.search_similar_chunks(question, top_k=top_k)
            
            if not relevant_chunks:
                return "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded relevant PDF documents."
            
            # Generate response using Mistral
            response = self.mistral_client.generate_response(question, relevant_chunks)
            
            logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question."
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index
        
        Returns:
            Dict containing index statistics
        """
        try:
            return self.pinecone_manager.get_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
