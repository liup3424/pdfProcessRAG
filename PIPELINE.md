# RAG Pipeline Documentation

This document explains the complete RAG (Retrieval-Augmented Generation) pipeline used in this system.

## Pipeline Overview

The RAG pipeline consists of 7 main stages:

```
PDF File → Extract Text → Chunk → Embed → Index → Retrieve → Re-rank → Generate Answer
```

## Stage 1: PDF Processing

**Module**: `pdf_processor.py`  
**Tool**: `PyMuPDFLoader` from LangChain

- **Input**: PDF file path
- **Output**: List of `Document` objects (one per page)
- **What it does**: Extracts text content from each page of the PDF

**Code**:
```python
pages = pdf_processor.load_pages(pdf_path)
```

## Stage 2: Text Chunking

**Module**: `chunker.py`  
**Tool**: `RecursiveCharacterTextSplitter` from LangChain with `tiktoken` for token counting

- **Input**: List of `Document` objects (pages)
- **Output**: List of text chunks with metadata
- **What it does**: 
  - Combines all pages into a single document
  - Splits text into chunks of 500 tokens with 50 token overlap
  - Preserves metadata (source file, page numbers)

**Configuration**:
- `CHUNK_SIZE = 500` tokens
- `CHUNK_OVERLAP = 50` tokens

**Code**:
```python
chunks = chunker.chunk_documents(pages)
```

## Stage 3: Embedding Generation

**Module**: `embedding.py`  
**Function**: `local_embedding()`

- **Input**: List of text strings
- **Output**: List of embedding vectors (1024 dimensions)
- **What it does**:
  - Calls external embedding API (`EMBEDDING_URL`)
  - Generates dense vector embeddings for each text chunk
  - Uses model: `qwen3-embedding-0.6b` (1024 dimensions)
  - Processes in batches for efficiency

**Configuration**:
- `EMBEDDING_URL`: API endpoint
- `EMBEDDING_DIMENSION = 1024`

**Code**:
```python
texts = [chunk["text"] for chunk in chunks]
embeddings = local_embedding(texts)
```

## Stage 4: Elasticsearch Indexing

**Module**: `es_indexer.py`  
**Tool**: Elasticsearch

- **Input**: Chunks with text + embeddings
- **Output**: Documents indexed in Elasticsearch
- **What it does**:
  - Creates index with mappings for:
    - Text content (for BM25 search)
    - Dense vector embeddings (for vector search)
    - Metadata (file name, page numbers, chunk IDs)
  - Stores each chunk as a document with both text and vector

**Configuration**:
- `ELASTICSEARCH_HOST`, `ELASTICSEARCH_USER`, `ELASTICSEARCH_PASSWORD`
- `ELASTICSEARCH_INDEX_NAME = "pdf_rag_index"`

**Code**:
```python
indexer.create_index()
indexer.index_documents(chunks, embeddings)
```

## Stage 5: Hybrid Retrieval

**Module**: `retriever.py`  
**Tool**: Elasticsearch hybrid search

- **Input**: Query text + query embedding
- **Output**: List of retrieved documents with scores
- **What it does**:
  - Performs hybrid search combining:
    - **BM25** (keyword-based search): 30% weight
    - **Vector similarity** (semantic search): 70% weight
  - Returns top K documents ranked by combined score

**Configuration**:
- `RETRIEVAL_TOP_K = 10` (number of documents to retrieve)
- `BM25_WEIGHT = 0.3`
- `VECTOR_WEIGHT = 0.7`

**Code**:
```python
query_embedding = local_embedding([query])[0]
results = retriever.search(query, query_embedding, top_k=10)
```

## Stage 6: Re-ranking

**Module**: `reranker.py`  
**Tool**: External re-ranker API (with RRF fallback)

- **Input**: Query + retrieved documents
- **Output**: Re-ranked list of documents
- **What it does**:
  - Calls external re-ranker API (`RERANK_URL`)
  - Uses model: `qwen3-reranker-0.6b`
  - Re-ranks documents based on relevance to query
  - Falls back to RRF (Reciprocal Rank Fusion) if API fails

**Configuration**:
- `RERANK_URL`: API endpoint
- `RERANK_TOP_K = 10` (number of documents to return after re-ranking)

**Code**:
```python
reranked = reranker.rerank(query, results, top_k=10)
```

## Stage 7: Answer Generation

**Module**: `answer_generator.py`  
**Tool**: LLM API (OpenAI, Anthropic, etc.)

- **Input**: Query + re-ranked documents
- **Output**: Generated answer with source citations
- **What it does**:
  - Builds context from retrieved documents
  - Creates prompt with query and context
  - Calls LLM API to generate answer
  - Extracts source citations
  - Falls back to simple template-based answers if LLM not configured

**Configuration**:
- `LLM_API_URL`: API endpoint (optional)
- `LLM_API_KEY`: API key (optional)
- `LLM_MODEL`: Model name (optional)

**Code**:
```python
answer_data = answer_generator.generate_answer_with_sources(query, reranked)
```

## Complete Pipeline Flow

### Processing a PDF:

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Process PDF (stages 1-4)
rag.process_pdf("document.pdf")
```

**What happens**:
1. Extract text from PDF → `pages`
2. Chunk pages → `chunks`
3. Generate embeddings → `embeddings`
4. Index in Elasticsearch → stored documents

### Querying:

```python
# Query (stages 5-7)
result = rag.query("What is the main topic?")
```

**What happens**:
1. Generate query embedding → `query_embedding`
2. Hybrid retrieval → `results` (10 documents)
3. Re-rank → `reranked` (10 documents)
4. Generate answer → `answer` with sources

## Configuration Summary

| Stage | Configuration | Default Value |
|-------|--------------|---------------|
| Chunking | `CHUNK_SIZE` | 500 tokens |
| Chunking | `CHUNK_OVERLAP` | 50 tokens |
| Embedding | `EMBEDDING_DIMENSION` | 1024 |
| Retrieval | `RETRIEVAL_TOP_K` | 10 |
| Retrieval | `BM25_WEIGHT` | 0.3 |
| Retrieval | `VECTOR_WEIGHT` | 0.7 |
| Re-ranking | `RERANK_TOP_K` | 10 |

## Error Handling

- **Embedding API fails**: Returns zero vectors (handled gracefully)
- **Re-ranker API fails**: Falls back to RRF (Reciprocal Rank Fusion)
- **LLM API fails**: Falls back to simple template-based answer generator
- **Elasticsearch connection fails**: Returns error message

## Performance Considerations

- **Batch processing**: Embeddings are generated in batches (default: 10 texts per batch)
- **Chunk size**: 500 tokens balances context and granularity
- **Hybrid search**: Combines keyword (BM25) and semantic (vector) search for better results
- **Re-ranking**: Refines results after initial retrieval for improved relevance

