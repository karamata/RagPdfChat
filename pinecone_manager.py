import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeManager:
    """
    Manages Pinecone vector database operations
    """
    
    def __init__(self, api_key: str, index_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Pinecone manager
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: SentenceTransformer model name
        """
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.index = None
        
    def initialize_index(self):
        """
        Initialize or connect to Pinecone index
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(1)
                    
            else:
                logger.info(f"Connecting to existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info("Successfully connected to Pinecone index")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise Exception(f"Failed to initialize Pinecone index: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def _generate_chunk_id(self, chunk_text: str, filename: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk
        
        Args:
            chunk_text: Text content of the chunk
            filename: Source filename
            chunk_index: Index of the chunk in the document
            
        Returns:
            str: Unique chunk ID
        """
        # Create a hash based on content and metadata
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        return f"{filename}_{chunk_index}_{content_hash}"
    
    def add_chunks(self, chunks: List[str], filename: str) -> bool:
        """
        Add text chunks to Pinecone index
        
        Args:
            chunks: List of text chunks
            filename: Source filename
            
        Returns:
            bool: Success status
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            if not chunks:
                logger.warning("No chunks to add")
                return True
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._generate_chunk_id(chunk, filename, i)
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "timestamp": time.time()
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully added {len(chunks)} chunks from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to Pinecone: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the index
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract relevant information
            results = []
            for match in search_results['matches']:
                result = {
                    "text": match['metadata']['text'],
                    "filename": match['metadata']['filename'],
                    "chunk_index": match['metadata']['chunk_index'],
                    "score": match['score']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index
        
        Returns:
            Dict containing index statistics
        """
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_index(self):
        """
        Delete the Pinecone index
        """
        try:
            if self.index_name in [index.name for index in self.pc.list_indexes()]:
                self.pc.delete_index(self.index_name)
                logger.info(f"Deleted index: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} does not exist")
                
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
