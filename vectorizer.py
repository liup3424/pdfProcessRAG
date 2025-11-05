"""
Vectorization Module
Generates vector embeddings for text chunks.
"""
from typing import List, Dict
import requests
import numpy as np
import config
import time


class Vectorizer:
    """Generate vector embeddings for text chunks."""
    
    def __init__(self, embedding_url: str = None):
        """
        Initialize vectorizer.
        
        Args:
            embedding_url: URL of the embedding API endpoint
        """
        self.embedding_url = embedding_url or config.EMBEDDING_URL
        self.dimension = config.EMBEDDING_DIMENSION
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else None
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_embedding_api(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call the embedding API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Prepare request payload
            # The API format may vary, adjust based on actual API specification
            payload = {
                "input": texts,
                "model": config.EMBEDDING_MODEL
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if "data" in result:
                # OpenAI-compatible format
                embeddings = [item["embedding"] for item in result["data"]]
            elif "embeddings" in result:
                # Alternative format
                embeddings = result["embeddings"]
            elif isinstance(result, list):
                # Direct list of embeddings
                embeddings = result
            else:
                raise ValueError(f"Unexpected API response format: {result}")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding API: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in texts]
        except Exception as e:
            print(f"Error processing embeddings: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector to unit length.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Normalized embedding vector
        """
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return embedding
        return (vec / norm).tolist()

