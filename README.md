# PDF Process RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for processing PDF files and answering questions based on their content.

## Features

- **PDF Processing**: Extract text from PDF files using pdfplumber and PyPDF2
- **Content Chunking**: Split extracted content into retrievable chunks
- **Vectorization**: Generate embeddings using qwen3-embedding-0.6b model
- **Elasticsearch Indexing**: Store content and vectors in Elasticsearch for efficient retrieval
- **Hybrid Search**: Combine BM25 keyword search and vector similarity search
- **Re-ranking**: Apply Reciprocal Rank Fusion (RRF) or re-ranker model (qwen3-reranker-0.6b)
- **Answer Generation**: Generate answers using LLM based on retrieved documents

## Requirements

- Python 3.8+
- Elasticsearch 8.x (local deployment)
- Access to embedding API (http://test.2brain.cn:9800/v1/emb)
- Access to re-ranker API (http://test.2brain.cn:2260/rerank)
- Optional: LLM API for answer generation

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd pdfProcessRAG
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Elasticsearch:**
   - Follow the Elasticsearch quickstart guide to set up a local instance
   - The quickstart script will generate a `.env` file with credentials
   - Extract the `ELASTICSEARCH_PASSWORD` and `ELASTICSEARCH_API_KEY` from the `.env` file

4. **Configure environment variables:**
   - Create a `.env` file in the project root
   - Copy the following template and fill in your values:
     ```
     ELASTICSEARCH_HOST=https://localhost:9200
     ELASTICSEARCH_USER=elastic
     ELASTICSEARCH_PASSWORD=your_password_here
     ELASTICSEARCH_API_KEY=your_api_key_here
     ELASTICSEARCH_VERIFY_CERTS=false
     
     EMBEDDING_URL=http://test.2brain.cn:9800/v1/emb
     RERANK_URL=http://test.2brain.cn:2260/rerank
     
     # Optional: LLM API for answer generation
     LLM_API_URL=
     LLM_MODEL=gpt-3.5-turbo
     ```

## Usage

### Command Line Interface

1. **Test Elasticsearch connection:**
   ```bash
   python main.py test
   ```

2. **Process a PDF file:**
   ```bash
   python main.py process --pdf path/to/your/document.pdf
   ```

3. **Query the system:**
   ```bash
   python main.py query --query "Your question here"
   ```

4. **Query with custom top-k:**
   ```bash
   python main.py query --query "Your question here" --top-k 10
   ```

### Python API

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()

# Process PDF files
rag.process_documents(["document1.pdf", "document2.pdf"])

# Query the system
result = rag.query("What is the main topic of the document?")
print(result["answer"])
print(f"Sources: {result['num_sources']}")
```

## Project Structure

```
pdfProcessRAG/
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF text extraction
├── chunker.py             # Text chunking
├── vectorizer.py          # Embedding generation
├── es_indexer.py          # Elasticsearch indexing
├── retriever.py           # Hybrid search retrieval
├── reranker.py            # Result re-ranking
├── answer_generator.py    # Answer generation
├── main.py                # CLI entry point
├── rag_pipeline.py        # Complete RAG pipeline
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Configuration

All configuration is managed through `config.py` and environment variables in `.env`. Key settings:

- **Chunking**: `CHUNK_SIZE` (default: 500), `CHUNK_OVERLAP` (default: 50)
- **Retrieval**: `RETRIEVAL_TOP_K` (default: 20)
- **Re-ranking**: `RERANK_TOP_K` (default: 10)
- **Hybrid Search Weights**: `BM25_WEIGHT` (default: 0.5), `VECTOR_WEIGHT` (default: 0.5)

## Workflow

1. **PDF Processing**: Extract text from PDF files
2. **Chunking**: Split text into manageable chunks with overlap
3. **Vectorization**: Generate embeddings for each chunk
4. **Indexing**: Store chunks and embeddings in Elasticsearch
5. **Retrieval**: Perform hybrid search (BM25 + vector) on queries
6. **Re-ranking**: Refine results using RRF or re-ranker model
7. **Answer Generation**: Generate final answer from top-ranked documents

## Notes

- The system uses qwen3-embedding-0.6b for embeddings (512 dimensions)
- The system uses qwen3-reranker-0.6b for re-ranking
- If LLM API is not configured, the system will return simple template-based answers
- Elasticsearch index is automatically created on first use
- The index name is configurable via `ELASTICSEARCH_INDEX_NAME` in config

## Troubleshooting

1. **Elasticsearch Connection Error:**
   - Verify Elasticsearch is running: `curl -k https://localhost:9200`
   - Check credentials in `.env` file
   - Ensure `ELASTICSEARCH_VERIFY_CERTS` is set correctly

2. **Embedding API Error:**
   - Verify the embedding API is accessible
   - Check network connectivity
   - The system will use zero vectors as fallback

3. **Re-ranker API Error:**
   - If API fails, the system automatically falls back to RRF

## License

This project is provided as-is for educational and research purposes.

