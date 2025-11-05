# PDF Process RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for processing PDF files and answering questions based on their content.

## Features

- **PDF Processing**: Extract text from PDF files using PyMuPDFLoader (LangChain)
- **Content Chunking**: Split extracted content into retrievable chunks using RecursiveCharacterTextSplitter with tiktoken
- **Vectorization**: Generate embeddings using qwen3-embedding-0.6b model (1024 dimensions)
- **Elasticsearch Indexing**: Store content and vectors in Elasticsearch for efficient retrieval
- **Hybrid Search**: Combine BM25 keyword search (30%) and vector similarity search (70%)
- **Re-ranking**: Apply re-ranker model (qwen3-reranker-0.6b) with RRF fallback
- **Answer Generation**: Generate answers using LLM (OpenAI, Anthropic, etc.) based on retrieved documents

## Requirements

- Python 3.8+
- Elasticsearch 8.x (local deployment)
- Access to embedding API 
- Access to re-ranker API 
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
   - Copy the example file: `cp .env.example .env`
   - Edit `.env` and fill in your values:
     - **REQUIRED**: Elasticsearch credentials (from Elasticsearch quickstart .env file)
     - **REQUIRED**: Embedding API URL
     - **REQUIRED**: Re-ranking API URL
     - **OPTIONAL**: LLM API URL and key for answer generation
   
   See `.env.example` for detailed configuration instructions.

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
from src.rag_pipeline import RAGPipeline

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
├── src/                   # Source code package
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── pdf_processor.py   # PDF text extraction (PyMuPDFLoader)
│   ├── chunker.py         # Text chunking (RecursiveCharacterTextSplitter)
│   ├── embedding.py       # Embedding generation (local_embedding function)
│   ├── es_indexer.py      # Elasticsearch indexing
│   ├── retriever.py       # Hybrid search retrieval (BM25 + vector)
│   ├── reranker.py        # Result re-ranking (API + RRF fallback)
│   ├── answer_generator.py # Answer generation (LLM API)
│   ├── rag_pipeline.py    # Complete RAG pipeline orchestration
│   └── setup_logging.py   # Logging configuration
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
├── PIPELINE.md            # Detailed pipeline documentation
├── SETUP.md               # Setup instructions
├── ENV_SETUP.md           # Environment variables setup guide
└── PDFtest/               # Test PDF files
    └── FixedIncomeChap1.pdf
```

## Configuration

All configuration is managed through `config.py` and environment variables in `.env`. Key settings:

- **Chunking**: `CHUNK_SIZE` (default: 500 tokens), `CHUNK_OVERLAP` (default: 50 tokens)
- **Retrieval**: `RETRIEVAL_TOP_K` (default: 10)
- **Re-ranking**: `RERANK_TOP_K` (default: 10)
- **Hybrid Search Weights**: `BM25_WEIGHT` (default: 0.3), `VECTOR_WEIGHT` (default: 0.7)
- **Embedding**: `EMBEDDING_DIMENSION` (1024 for qwen3-embedding-0.6b)

## Pipeline

The RAG pipeline consists of 7 stages:

1. **PDF Processing**: Extract text from PDF files using PyMuPDFLoader
2. **Chunking**: Split text into 500-token chunks with 50-token overlap using RecursiveCharacterTextSplitter
3. **Embedding**: Generate 1024-dimensional embeddings using qwen3-embedding-0.6b API
4. **Indexing**: Store chunks and embeddings in Elasticsearch with hybrid search support
5. **Retrieval**: Perform hybrid search (30% BM25 + 70% vector) to retrieve top 10 documents
6. **Re-ranking**: Re-rank results using qwen3-reranker-0.6b API (with RRF fallback)
7. **Answer Generation**: Generate final answer using LLM based on top-ranked documents

See [PIPELINE.md](PIPELINE.md) for detailed documentation of each stage.

## Notes

- The system uses **qwen3-embedding-0.6b** for embeddings (1024 dimensions)
- The system uses **qwen3-reranker-0.6b** for re-ranking
- If LLM API is not configured, the system will return simple template-based answers
- Elasticsearch index is automatically created on first use
- The index name is configurable via `ELASTICSEARCH_INDEX_NAME` in config
- Chunking uses **tiktoken** for accurate token counting
- Hybrid search combines keyword (BM25) and semantic (vector) search for better results

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

