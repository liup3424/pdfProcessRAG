"""
Elasticsearch Indexing Module
Stores content and vectors in Elasticsearch.
"""
from typing import List, Dict, Optional
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from . import config
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
            logging.getLogger(__name__).info("Index '%s' already exists.", self.index_name)
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
            logging.getLogger(__name__).info("Index '%s' created successfully.", self.index_name)
            return True
        except Exception as e:
            # Fallback to body parameter for older versions
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                logging.getLogger(__name__).info("Index '%s' created successfully.", self.index_name)
                return True
            except Exception as e2:
                logging.getLogger(__name__).error("Error creating index: %s", e2, exc_info=True)
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
                logging.getLogger(__name__).warning("%d documents failed to index.", len(failed))
                # Print first few errors for debugging
                for i, error in enumerate(failed[:3]):
                    logging.getLogger(__name__).warning("  Error %d: %s", i+1, error)
                return False
            
            logging.getLogger(__name__).info("Successfully indexed %d documents.", success)
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error("Error indexing documents: %s", e, exc_info=True)
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
                logging.getLogger(__name__).info("Index '%s' deleted successfully.", self.index_name)
                return True
            else:
                logging.getLogger(__name__).info("Index '%s' does not exist.", self.index_name)
                return False
        except Exception as e:
            logging.getLogger(__name__).error("Error deleting index: %s", e, exc_info=True)
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
            logging.getLogger(__name__).error("Error getting index stats: %s", e, exc_info=True)
            return {}
    
    def test_connection(self) -> bool:
        """
        Test connection to Elasticsearch.
        
        Returns:
            True if connection is successful
        """
        try:
            info = self.client.info()
            logging.getLogger(__name__).info("Connected to Elasticsearch: %s", info['version']['number'])
            return True
        except Exception as e:
            logging.getLogger(__name__).error("Error connecting to Elasticsearch: %s", e, exc_info=True)
            return False

