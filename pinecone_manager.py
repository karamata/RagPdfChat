import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import logging
import time
import json

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
        
        # Try to initialize Pinecone and SentenceTransformer
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.pc = Pinecone(api_key=api_key)
            self.ServerlessSpec = ServerlessSpec
            self.pinecone_available = True
        except ImportError:
            logger.warning("Pinecone library not available. Using fallback mode.")
            self.pc = None
            self.ServerlessSpec = None
            self.pinecone_available = False
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.sentence_transformers_available = True
        except ImportError:
            logger.warning("SentenceTransformers library not available. Using fallback mode.")
            self.embedding_model = None
            self.sentence_transformers_available = False
        
        self.index = None
        self.local_storage = {}  # Fallback storage
        
    def initialize_index(self):
        """
        Initialize or connect to Pinecone index
        """
        if not self.pinecone_available:
            logger.info("Using local storage fallback instead of Pinecone")
            return
            
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
                    spec=self.ServerlessSpec(
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
        if not self.sentence_transformers_available:
            # Simple fallback: use hash-based pseudo-embeddings
            logger.warning("Using fallback hash-based embeddings")
            embeddings = []
            for text in texts:
                # Create a simple hash-based embedding
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                # Convert to float vector
                embedding = [float(b) / 255.0 for b in hash_bytes]
                # Pad or truncate to desired dimension
                while len(embedding) < self.embedding_dimension:
                    embedding.extend(embedding[:min(len(embedding), self.embedding_dimension - len(embedding))])
                embedding = embedding[:self.embedding_dimension]
                embeddings.append(embedding)
            return embeddings
            
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
            if not chunks:
                logger.warning("No chunks to add")
                return True
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.generate_embeddings(chunks)
            
            if not self.pinecone_available or not self.index:
                # Use local storage fallback
                logger.info("Using local storage fallback for chunks")
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = self._generate_chunk_id(chunk, filename, i)
                    self.local_storage[chunk_id] = {
                        "text": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "timestamp": time.time(),
                        "embedding": embedding
                    }
                logger.info(f"Successfully stored {len(chunks)} chunks locally")
                return True
            
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
            logger.error(f"Error adding chunks: {str(e)}")
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
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            if not self.pinecone_available or not self.index:
                # Use local storage fallback with simple similarity
                logger.info("Using local storage search fallback")
                results = []
                
                for chunk_id, chunk_data in self.local_storage.items():
                    # Simple cosine similarity calculation
                    chunk_embedding = chunk_data["embedding"]
                    similarity = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
                    
                    result = {
                        "text": chunk_data["text"],
                        "filename": chunk_data["filename"],
                        "chunk_index": chunk_data["chunk_index"],
                        "score": similarity
                    }
                    results.append(result)
                
                # Sort by similarity score and return top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:top_k]
                
                logger.info(f"Found {len(results)} similar chunks using local search")
                return results
            
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
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0
    
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
