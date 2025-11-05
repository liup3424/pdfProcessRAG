"""Retrieval Module - Hybrid BM25 + Vector Search."""
from typing import List, Dict, Optional
import logging
from elasticsearch import Elasticsearch
from . import config
import numpy as np


class HybridRetriever:
    """Perform hybrid search combining BM25 and vector search."""
    
    def __init__(self, es_client: Elasticsearch = None):
        """
        Initialize retriever.
        
        Args:
            es_client: Elasticsearch client instance
        """
        if es_client:
            self.client = es_client
        else:
            self.client = self._create_client()
        self.index_name = config.ELASTICSEARCH_INDEX_NAME
        self.bm25_weight = config.BM25_WEIGHT
        self.vector_weight = config.VECTOR_WEIGHT
    
    def _create_client(self) -> Elasticsearch:
        """Create Elasticsearch client using available credentials."""
        if config.ELASTICSEARCH_API_KEY:
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                api_key=config.ELASTICSEARCH_API_KEY,
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        elif config.ELASTICSEARCH_PASSWORD:
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                basic_auth=(config.ELASTICSEARCH_USER, config.ELASTICSEARCH_PASSWORD),
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        else:
            es_client = Elasticsearch(
                [config.ELASTICSEARCH_HOST],
                verify_certs=config.ELASTICSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
        return es_client
    
    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = None,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Text query for BM25 search
            query_embedding: Query embedding for vector search
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of search results with scores
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        # Build the search query
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # BM25 keyword search
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "boost": self.bm25_weight
                                }
                            }
                        },
                        # Vector similarity search
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": """
                                        double similarity = cosineSimilarity(params.query_vector, 'embedding');
                                        if (Double.isNaN(similarity) || Double.isInfinite(similarity)) {
                                            return 0.0;
                                        }
                                        return similarity + 1.0;
                                    """,
                                    "params": {
                                        "query_vector": query_embedding
                                    }
                                },
                                "boost": self.vector_weight
                            }
                        }
                    ]
                }
            },
            "_source": ["text", "chunk_id", "metadata"],
            "min_score": 0.1  # Minimum score threshold
        }
        
        # Add filters if provided
        if filters:
            search_body["query"]["bool"]["filter"] = filters
        
        try:
            # Use query parameter for newer Elasticsearch client versions
            try:
                response = self.client.search(
                    index=self.index_name,
                    query=search_body["query"],
                    size=search_body.get("size", 10),
                    _source=search_body.get("_source", True),
                    min_score=search_body.get("min_score", 0)
                )
            except TypeError:
                # Fallback to body parameter for older versions
                response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "id": hit["_id"]
                })
            
            return results
            
        except Exception as e:
            logging.getLogger(__name__).error("Error performing search: %s", e, exc_info=True)
            return []
    
    def search_bm25_only(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Perform BM25 keyword search only.
        
        Args:
            query: Text query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        search_body = {
            "size": top_k,
            "query": {
                "match": {
                    "text": query
                }
            },
            "_source": ["text", "chunk_id", "metadata"]
        }
        
        try:
            # Use query parameter for newer Elasticsearch client versions
            try:
                response = self.client.search(
                    index=self.index_name,
                    query=search_body["query"],
                    size=search_body.get("size", 10),
                    _source=search_body.get("_source", True)
                )
            except TypeError:
                # Fallback to body parameter for older versions
                response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "id": hit["_id"]
                })
            
            return results
            
        except Exception as e:
            logging.getLogger(__name__).error("Error performing BM25 search: %s", e, exc_info=True)
            return []
    
    def search_vector_only(
        self,
        query_embedding: List[float],
        top_k: int = None
    ) -> List[Dict]:
        """
        Perform vector similarity search only.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            },
            "_source": ["text", "chunk_id", "metadata"],
            "min_score": 0.1
        }
        
        try:
            # Use query parameter for newer Elasticsearch client versions
            try:
                response = self.client.search(
                    index=self.index_name,
                    query=search_body["query"],
                    size=search_body.get("size", 10),
                    _source=search_body.get("_source", True),
                    min_score=search_body.get("min_score", 0)
                )
            except TypeError:
                # Fallback to body parameter for older versions
                response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "id": hit["_id"]
                })
            
            return results
            
        except Exception as e:
            logging.getLogger(__name__).error("Error performing vector search: %s", e, exc_info=True)
            return []

