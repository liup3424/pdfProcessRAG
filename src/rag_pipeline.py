"""
RAG Pipeline Module
Complete pipeline for processing PDFs and querying.
"""
import logging
from typing import List, Dict, Optional
from .pdf_processor import PDFProcessor
from .chunker import TextChunker
from .es_indexer import ESIndexer
from .retriever import HybridRetriever
from .reranker import Reranker
from .answer_generator import AnswerGenerator
from . import config
from .embedding import local_embedding

# Set up logger
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for document processing and querying."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.indexer = ESIndexer()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.answer_generator = AnswerGenerator()
    
    def process_documents(self, pdf_paths: List[str]) -> Dict[str, bool]:
        """Process multiple PDF files and return success status per file."""
        results: Dict[str, bool] = {}
        for pdf_path in pdf_paths:
            logger.info("Processing PDF: %s", pdf_path)
            success = self.process_pdf(pdf_path)
            results[pdf_path] = success
        return results
    
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a single PDF file: extract, chunk, vectorize, and index.
        Uses the new approach: PyMuPDFLoader -> RecursiveCharacterTextSplitter with tiktoken -> local_embedding.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if processing was successful
        """
        logger.info("Processing single PDF: %s", pdf_path)
        
        # Step 1: Load PDF pages using PyMuPDFLoader
        logger.info("Step 1: Loading PDF pages")
        try:
            pages = self.pdf_processor.load_pages(pdf_path)
            logger.info("Loaded %d pages from PDF", len(pages))
        except Exception as e:
            logger.error("Error loading PDF: %s", e, exc_info=True)
            return False
        
        # Step 2: Chunk the pages using RecursiveCharacterTextSplitter with tiktoken
        logger.info("Step 2: Chunking pages")
        logger.info("Starting chunking process")
        try:
            chunks = self.chunker.chunk_documents(pages)
            logger.info(f"Chunking completed: {len(chunks)} chunks created")
        except Exception as e:
            logger.error(f"Chunking failed: {e}", exc_info=True)
            return False
        
        # Step 3: Generate embeddings using local_embedding
        logger.info("Step 3: Generating embeddings")
        logger.info("Starting embedding generation")
        try:
            texts = [chunk["text"] for chunk in chunks]
            logger.info(f"Preparing {len(texts)} texts for embedding")
            embeddings = local_embedding(texts)
            logger.info(f"Embedding generation completed: {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return False
        
        # Step 4: Index in Elasticsearch
        logger.info("Step 4: Indexing in Elasticsearch")
        try:
            # Ensure index exists
            self.indexer.create_index()
            
            # Index documents
            success = self.indexer.index_documents(chunks, embeddings)
            if success:
                logger.info("Successfully indexed %d documents", len(chunks))
            else:
                logger.error("Error indexing documents")
                return False
        except Exception as e:
            logger.error("Error indexing: %s", e, exc_info=True)
            return False
        
        logger.info("PDF processing completed successfully")
        return True
    
    
    
    def query(
        self,
        query: str,
        top_k_retrieval: int = None,
        top_k_rerank: int = None
    ) -> Dict:
        """
        Query the RAG system with full pipeline.
        
        Args:
            query: Query text
            top_k_retrieval: Number of documents to retrieve
            top_k_rerank: Number of documents to return after re-ranking
            
        Returns:
            Query result with answer and sources
        """
        try:
            # Generate query embedding using local_embedding
            query_embeddings = local_embedding([query])
            query_embedding = query_embeddings[0] if query_embeddings else None
            
            if not query_embedding or all(v == 0.0 for v in query_embedding):
                logger.error("Query embedding generation failed or returned zero vector")
                return {"error": "Failed to generate query embedding"}
            
            # Retrieve
            results = self.retriever.search(
                query,
                query_embedding,
                top_k=top_k_retrieval or config.RETRIEVAL_TOP_K
            )
            
            # Re-rank
            reranked = self.reranker.rerank(
                query,
                results,
                top_k=top_k_rerank or config.RERANK_TOP_K
            )
            
            # Generate answer
            answer_data = self.answer_generator.generate_answer_with_sources(
                query,
                reranked
            )
            
            return answer_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def initialize_index(self):
        """Initialize Elasticsearch index."""
        self.indexer.create_index()
        logger.info("Index initialized successfully")
    
    def test_connection(self) -> bool:
        """Test Elasticsearch connection."""
        return self.indexer.test_connection()

