"""
Elasticsearch Indexing Module
Stores content and vectors in Elasticsearch.
"""
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import config
import json


class ESIndexer:
    """Index documents and vectors in Elasticsearch."""
    
    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = self._create_client()
        self.index_name = config.ELASTICSEARCH_INDEX_NAME
    
    def _create_client(self) -> Elasticsearch:
        """Create and configure Elasticsearch client."""
        # Determine authentication method
        if config.ELASTICSEARCH_API_KEY:
            # Use API key authentication
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                api_key=config.ELASTICSEARCH_API_KEY,
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        elif config.ELASTICSEARCH_PASSWORD:
            # Use basic authentication
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                basic_auth=(config.ELASTICSEARCH_USER, config.ELASTICSEARCH_PASSWORD),
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        else:
            # No authentication
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        
        return es_client
    
    def create_index(self, dimension: int = None) -> bool:
        """
        Create Elasticsearch index with mapping for hybrid search.
        
        Args:
            dimension: Dimension of the embedding vectors
            
        Returns:
            True if index was created successfully
        """
        dimension = dimension or config.EMBEDDING_DIMENSION
        
        # Check if index already exists
        if self.client.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' already exists.")
            return True
        
        # Define index mapping with dense_vector for embeddings
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "chunk_id": {
                        "type": "integer"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "text"},
                            "author": {"type": "text"},
                            "file_name": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "total_pages": {"type": "integer"},
                            "page_number": {"type": "integer"}
                        }
                    }
                }
            }
        }
        
        try:
            # Use mappings parameter for newer Elasticsearch client versions
            self.client.indices.create(
                index=self.index_name,
                mappings=mapping["mappings"]
            )
            print(f"Index '{self.index_name}' created successfully.")
            return True
        except Exception as e:
            # Fallback to body parameter for older versions
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                print(f"Index '{self.index_name}' created successfully.")
                return True
            except Exception as e2:
                print(f"Error creating index: {e2}")
                return False
    
    def index_documents(self, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Index documents with their embeddings.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors corresponding to chunks
            
        Returns:
            True if indexing was successful
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Ensure index exists
        self.create_index()
        
        # Prepare documents for bulk indexing
        actions = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "_index": self.index_name,
                "_source": {
                    "text": chunk["text"],
                    "embedding": embedding,
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk.get("metadata", {})
                }
            }
            actions.append(doc)
        
        try:
            # Bulk index documents
            success, failed = bulk(self.client, actions, chunk_size=100, raise_on_error=False)
            
            if failed:
                print(f"Warning: {len(failed)} documents failed to index.")
                # Print first few errors for debugging
                for i, error in enumerate(failed[:3]):
                    print(f"  Error {i+1}: {error}")
                return False
            
            print(f"Successfully indexed {success} documents.")
            return True
            
        except Exception as e:
            print(f"Error indexing documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the Elasticsearch index.
        
        Returns:
            True if deletion was successful
        """
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"Index '{self.index_name}' deleted successfully.")
                return True
            else:
                print(f"Index '{self.index_name}' does not exist.")
                return False
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test connection to Elasticsearch.
        
        Returns:
            True if connection is successful
        """
        try:
            info = self.client.info()
            print(f"Connected to Elasticsearch: {info['version']['number']}")
            return True
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            return False

