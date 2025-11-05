"""
Re-ranking Module
Applies Reciprocal Rank Fusion (RRF) or re-ranker model to refine search results.
"""
from typing import List, Dict, Optional
import logging
import requests
from . import config
from collections import defaultdict


class Reranker:
    """Re-rank search results using RRF or re-ranker model."""
    
    def __init__(self, rerank_url: str = None, use_reranker_api: bool = True):
        """
        Initialize reranker.
        
        Args:
            rerank_url: URL of the re-ranker API endpoint
            use_reranker_api: If True, use re-ranker API, otherwise use RRF
        """
        self.rerank_url = rerank_url or config.RERANK_URL
        self.use_reranker_api = use_reranker_api
        self.top_k = config.RERANK_TOP_K
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        Re-rank search results.
        
        Args:
            query: Original query text
            results: List of search results to re-rank
            top_k: Number of top results to return after re-ranking
            
        Returns:
            Re-ranked list of results
        """
        top_k = top_k or self.top_k
        
        if not results:
            return []
        
        if self.use_reranker_api:
            return self._rerank_with_api(query, results, top_k)
        else:
            return self._rerank_with_rrf(results, top_k)
    
    def _rerank_with_api(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Re-rank using re-ranker API.
        
        Args:
            query: Query text
            results: List of results to re-rank
            top_k: Number of top results
            
        Returns:
            Re-ranked results
        """
        try:
            # Prepare documents for re-ranking
            documents = [result["text"] for result in results]
            
            # Call re-ranker API
            payload = {
                "query": query,
                "documents": documents,
                "model": config.RERANK_MODEL,
                "top_k": top_k
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.rerank_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Debug: Log the response structure
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Reranker API Response keys: {list(result.keys()) if isinstance(result, dict) else 'list'}")
            if isinstance(result, dict) and "results" in result:
                logger.debug(f"Results type: {type(result['results'])}, first item keys: {list(result['results'][0].keys()) if result['results'] and isinstance(result['results'][0], dict) else 'N/A'}")
            
            # Handle different response formats
            reranked_indices = []
            reranked_scores = {}
            
            if "ranked_documents" in result and isinstance(result["ranked_documents"], list):
                # Format: {"ranked_documents": [{"document": "...", "score": 0.9, "index": 0}, ...], "scores": [...]}
                for item in result["ranked_documents"]:
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("rank", len(reranked_indices)))
                        score = item.get("score", item.get("relevance_score", 0.0))
                        reranked_indices.append(idx)
                        reranked_scores[idx] = score
            elif "results" in result and isinstance(result["results"], list):
                # Format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
                for item in result["results"]:
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("rank", len(reranked_indices)))
                        score = item.get("relevance_score", item.get("score", 0.0))
                        reranked_indices.append(idx)
                        reranked_scores[idx] = score
            elif "data" in result and isinstance(result["data"], list):
                # Format: {"data": [{"index": 0, "score": 0.9}, ...]}
                for item in result["data"]:
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("rank", len(reranked_indices)))
                        score = item.get("score", item.get("relevance_score", 0.0))
                        reranked_indices.append(idx)
                        reranked_scores[idx] = score
            elif isinstance(result, list):
                # Format: [{"index": 0, "score": 0.9}, ...]
                for item in result:
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("rank", len(reranked_indices)))
                        score = item.get("score", item.get("relevance_score", item.get("relevance_score", 0.0)))
                        reranked_indices.append(idx)
                        reranked_scores[idx] = score
            elif "result" in result:
                # Try nested result structure
                nested_result = result["result"]
                if isinstance(nested_result, list):
                    for item in nested_result:
                        if isinstance(item, dict):
                            idx = item.get("index", item.get("rank", len(reranked_indices)))
                            score = item.get("score", item.get("relevance_score", 0.0))
                            reranked_indices.append(idx)
                            reranked_scores[idx] = score
                elif isinstance(nested_result, dict) and "results" in nested_result:
                    for item in nested_result["results"]:
                        if isinstance(item, dict):
                            idx = item.get("index", item.get("rank", len(reranked_indices)))
                            score = item.get("relevance_score", item.get("score", 0.0))
                            reranked_indices.append(idx)
                            reranked_scores[idx] = score
            
            if not reranked_indices:
                # Fallback to RRF if API format is unexpected
                logger.warning(f"Unexpected reranker API response format. Keys: {list(result.keys()) if isinstance(result, dict) else 'list'}")
                logger.debug(f"Full response: {result}")
                return self._rerank_with_rrf(results, top_k)
            
            # Re-order results based on re-ranker scores
            reranked_results = []
            for idx in reranked_indices[:top_k]:
                if idx < len(results):
                    result = results[idx].copy()
                    result["rerank_score"] = reranked_scores.get(idx, 0.0)
                    reranked_results.append(result)
            
            return reranked_results
            
        except requests.exceptions.RequestException as e:
            logging.getLogger(__name__).error("Error calling re-ranker API: %s", e)
            logging.getLogger(__name__).info("Falling back to RRF")
            return self._rerank_with_rrf(results, top_k)
        except Exception as e:
            logging.getLogger(__name__).error("Error processing re-ranking: %s", e, exc_info=True)
            return self._rerank_with_rrf(results, top_k)
    
    def _rerank_with_rrf(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-rank using Reciprocal Rank Fusion (RRF).
        
        Args:
            results: List of results to re-rank
            top_k: Number of top results
            
        Returns:
            Re-ranked results
        """
        # RRF formula: score = sum(1 / (k + rank))
        k = 60  # RRF constant
        
        # Calculate RRF scores
        rrf_scores = {}
        for rank, result in enumerate(results, 1):
            doc_id = result.get("id", str(rank))
            rrf_score = 1.0 / (k + rank)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(
            results,
            key=lambda x: rrf_scores.get(x.get("id", ""), 0.0),
            reverse=True
        )
        
        # Add RRF scores to results
        for result in sorted_results:
            doc_id = result.get("id", "")
            result["rerank_score"] = rrf_scores.get(doc_id, 0.0)
        
        return sorted_results[:top_k]
    
    def rerank_multiple_queries(
        self,
        queries: List[str],
        query_results: List[List[Dict]],
        top_k: int = None
    ) -> List[Dict]:
        """
        Re-rank results from multiple queries using RRF.
        
        Args:
            queries: List of query texts
            query_results: List of result lists, one per query
            top_k: Number of top results
            
        Returns:
            Re-ranked and merged results
        """
        top_k = top_k or self.top_k
        
        if not queries or not query_results:
            return []
        
        if len(queries) != len(query_results):
            raise ValueError("Number of queries must match number of result lists")
        
        # Combine results from all queries
        all_results = []
        for results in query_results:
            all_results.extend(results)
        
        # Use RRF to re-rank combined results
        return self._rerank_with_rrf(all_results, top_k)

