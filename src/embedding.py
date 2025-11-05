"""
Embedding Module
Provides local embedding function for vectorization.
"""
from typing import List
import logging
import requests
from . import config

# Set up logger
logger = logging.getLogger(__name__)


def local_embedding(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Generate embeddings for texts using the embedding API.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process in each API call
        
    Returns:
        List of embedding vectors (list of floats)
    """
    logger.info(f"Starting embedding generation: {len(texts)} texts")
    logger.info(f"Batch size: {batch_size}, API URL: {config.EMBEDDING_URL}")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} texts")
        
        batch_embeddings = _call_embedding_api(batch)
        all_embeddings.extend(batch_embeddings)
        
        logger.debug(f"Batch {batch_num} completed: {len(batch_embeddings)} embeddings generated")
    
    logger.info(f"Embedding generation complete: {len(all_embeddings)} embeddings, "
               f"dimension: {len(all_embeddings[0]) if all_embeddings else 0}")
    
    return all_embeddings


def _call_embedding_api(texts: List[str]) -> List[List[float]]:
    """
    Call the embedding API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        # Try with "texts" field (as API expects)
        payload = {
            "texts": texts,
            "model": config.EMBEDDING_MODEL
        }
        
        logger.debug(f"API request: {len(texts)} texts, model: {config.EMBEDDING_MODEL}")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            config.EMBEDDING_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        logger.debug(f"API response status: {response.status_code}")
        
        # If 422, try OpenAI-compatible format
        if response.status_code == 422:
            logger.warning(f"422 error with 'texts' format, trying OpenAI-compatible 'input' format")
            payload = {
                "input": texts,
                "model": config.EMBEDDING_MODEL
            }
            response = requests.post(
                config.EMBEDDING_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            logger.debug(f"API response status (input format): {response.status_code}")
        
        # If still 422, try individual text format
        if response.status_code == 422:
            logger.warning(f"422 error with batch format, trying individual text format")
            # Try alternative format - process texts one by one
            embeddings = []
            for idx, text in enumerate(texts):
                alt_payload = {"texts": [text], "model": config.EMBEDDING_MODEL}
                logger.debug(f"Processing text {idx+1}/{len(texts)} individually")
                alt_response = requests.post(
                    config.EMBEDDING_URL,
                    json=alt_payload,
                    headers=headers,
                    timeout=30
                )
                if alt_response.status_code == 200:
                    alt_result = alt_response.json()
                    # Handle response with text_vectors format
                    if "data" in alt_result and isinstance(alt_result["data"], dict) and "text_vectors" in alt_result["data"]:
                        text_vectors = alt_result["data"]["text_vectors"]
                        if text_vectors and len(text_vectors) > 0:
                            embeddings.append(text_vectors[0])
                            logger.debug(f"Text {idx+1}: Got embedding with {len(text_vectors[0])} dimensions from text_vectors")
                    elif "data" in alt_result and isinstance(alt_result["data"], list) and len(alt_result["data"]) > 0:
                        embeddings.append(alt_result["data"][0]["embedding"])
                        logger.debug(f"Text {idx+1}: Got embedding from data array")
                    elif "embedding" in alt_result:
                        embeddings.append(alt_result["embedding"])
                        logger.debug(f"Text {idx+1}: Got embedding with {len(alt_result['embedding'])} dimensions")
                    elif "embeddings" in alt_result and len(alt_result["embeddings"]) > 0:
                        embeddings.append(alt_result["embeddings"][0])
                        logger.debug(f"Text {idx+1}: Got embedding from embeddings array")
                    else:
                        raise ValueError(f"Unexpected API response: {alt_result}")
                else:
                    logger.error(f"API error for text {idx+1}: {alt_response.status_code} - {alt_response.text}")
                    raise requests.exceptions.RequestException(
                        f"API error: {alt_response.status_code} - {alt_response.text}"
                    )
            logger.info(f"Successfully generated {len(embeddings)} embeddings using individual format")
            return embeddings
        
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            logger.error(f"API returned non-JSON response: {response.text[:200]}")
            raise ValueError(f"API returned non-JSON response: {response.text[:200]}")
        
        logger.debug(f"API response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Handle different response formats
        if isinstance(result, dict):
            if "data" in result:
                data = result["data"]
                # Check if data is a dict with "text_vectors" (API format)
                if isinstance(data, dict) and "text_vectors" in data:
                    embeddings = data["text_vectors"]
                    logger.debug(f"Got embeddings from 'data.text_vectors' field: {len(embeddings)} embeddings")
                # Check if data is a list (OpenAI-compatible format)
                elif isinstance(data, list):
                    embeddings = [item["embedding"] for item in data]
                    logger.debug(f"Got embeddings from 'data' list: {len(embeddings)} embeddings")
                else:
                    logger.error(f"Unexpected 'data' format: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    raise ValueError(f"Unexpected 'data' format: {data}")
            elif "embeddings" in result:
                # Alternative format
                embeddings = result["embeddings"]
                logger.debug(f"Got embeddings from 'embeddings' field: {len(embeddings)} embeddings")
            elif "embedding" in result:
                # Single embedding (shouldn't happen with batch, but handle it)
                embeddings = [result["embedding"]]
                logger.debug(f"Got single embedding from 'embedding' field")
            else:
                logger.error(f"Unexpected API response format. Keys: {list(result.keys())}")
                raise ValueError(f"Unexpected API response format: {result}")
        elif isinstance(result, list):
            # Direct list of embeddings
            embeddings = result
            logger.debug(f"Got embeddings as direct list: {len(embeddings)} embeddings")
        else:
            logger.error(f"Unexpected API response type: {type(result)}")
            raise ValueError(f"Unexpected API response format: {result}")
        
        if embeddings:
            logger.info(f"Successfully generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
        
        return embeddings
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling embedding API: {e}")
        if 'response' in locals():
            logger.error(f"  Response status: {response.status_code}")
            logger.error(f"  Response text (first 500 chars): {response.text[:500]}")
        else:
            logger.error(f"  No response received")
        logger.warning(f"Returning zero vectors as fallback for {len(texts)} texts")
        # Return zero vectors as fallback
        return [[0.0] * config.EMBEDDING_DIMENSION for _ in texts]
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if 'response' in locals():
            try:
                logger.error(f"  Response status: {response.status_code}")
                logger.error(f"  Response text (first 500 chars): {response.text[:500]}")
            except:
                pass
        logger.warning(f"Returning zero vectors as fallback for {len(texts)} texts")
        return [[0.0] * config.EMBEDDING_DIMENSION for _ in texts]

