"""
Configuration file for the RAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Elasticsearch Configuration
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "https://localhost:9200")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "elastic")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ELASTICSEARCH_VERIFY_CERTS = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "pdf_rag_index")

# Embedding Configuration
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "")
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_DIMENSION = 1024  # Dimension for qwen3-embedding-0.6b

# Re-ranking Configuration
RERANK_URL = os.getenv("RERANK_URL", "")
RERANK_MODEL = "qwen3-reranker-0.6b"
RERANK_TOP_K = 10  # Number of results to re-rank

# Chunking Configuration
CHUNK_SIZE = 500  # Tokens per chunk (using tiktoken)
CHUNK_OVERLAP = 50  # Overlap between chunks in tokens

# Retrieval Configuration
RETRIEVAL_TOP_K = 10  # Number of documents to retrieve before re-ranking
BM25_WEIGHT = 0.3  # Weight for BM25 score in hybrid search
VECTOR_WEIGHT = 0.7  # Weight for vector score in hybrid search

# LLM Configuration (for answer generation)
LLM_API_URL = os.getenv("LLM_API_URL", "")  # Optional: set if you have an LLM API
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # API key for LLM (e.g., OpenAI API key)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")  # Model name for answer generation

