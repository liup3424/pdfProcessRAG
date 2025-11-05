"""
Content Chunking Module
Splits text into retrievable chunks using LangChain's RecursiveCharacterTextSplitter with tiktoken.
"""
from typing import List, Dict
import logging
import tiktoken
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document
from . import config

# Set up logger
logger = logging.getLogger(__name__)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a string using tiktoken.
    
    Args:
        string: The text string to count tokens for
        encoding_name: The encoding to use (default: cl100k_base for GPT models)
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class TextChunker:
    """Split text into chunks for retrieval using RecursiveCharacterTextSplitter with token counting."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, encoding_name: str = "cl100k_base"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Size of each chunk in tokens (default from config)
            chunk_overlap: Overlap between chunks in tokens (default from config)
            encoding_name: Token encoding to use (default: cl100k_base)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.encoding_name = encoding_name
        
        # Initialize RecursiveCharacterTextSplitter with token counting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=num_tokens_from_string,
            separators=["\n\n", "\n", ". ", " ", ""]  # Default separators
        )
    
    def chunk_documents(self, pages: List[Document]) -> List[Dict]:
        """
        Split documents (pages) into chunks using RecursiveCharacterTextSplitter.
        Combines all pages into one document before chunking to respect CHUNK_SIZE.
        
        Args:
            pages: List of Document objects from PyMuPDFLoader
            
        Returns:
            List of chunk dictionaries, each containing:
                - text: Chunk text
                - chunk_id: Unique chunk identifier
                - metadata: Chunk metadata
        """
        if not pages:
            logger.warning("No pages provided for chunking")
            return []
        
        logger.info(f"Starting chunking: {len(pages)} pages")
        logger.info(f"Chunk size: {self.chunk_size} tokens, Overlap: {self.chunk_overlap} tokens")
        
        # Calculate total tokens before chunking
        total_tokens = sum(num_tokens_from_string(page.page_content) for page in pages)
        logger.info(f"Total tokens in pages: {total_tokens}")
        
        # Combine all pages into one document for chunking
        # This ensures chunks respect CHUNK_SIZE across page boundaries
        combined_text = "\n\n".join([page.page_content for page in pages])
        
        # Get metadata from first page (for file-level metadata)
        combined_metadata = pages[0].metadata.copy() if pages[0].metadata else {}
        
        # Create a single combined document
        combined_doc = Document(
            page_content=combined_text,
            metadata=combined_metadata
        )
        
        # Split the combined document using chunk_size
        split_docs = self.text_splitter.split_documents([combined_doc])
        
        logger.info(f"Created {len(split_docs)} chunks from {len(pages)} pages (combined)")
        
        # Convert to the expected format
        chunks = []
        chunk_token_stats = []
        for chunk_id, split_doc in enumerate(split_docs):
            chunk_metadata = split_doc.metadata.copy() if split_doc.metadata else {}
            chunk_metadata["chunk_id"] = chunk_id
            
            chunk_text = split_doc.page_content.strip()
            chunk_tokens = num_tokens_from_string(chunk_text)
            chunk_token_stats.append(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "metadata": chunk_metadata
            })
            
            # Log first few chunks for debugging
            if chunk_id < 3:
                logger.debug(f"Chunk {chunk_id}: {chunk_tokens} tokens, {len(chunk_text)} chars")
        
        # Log chunk statistics
        if chunk_token_stats:
            avg_tokens = sum(chunk_token_stats) / len(chunk_token_stats)
            min_tokens = min(chunk_token_stats)
            max_tokens = max(chunk_token_stats)
            logger.info(f"Chunk token statistics - Avg: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
            logger.info(f"Expected chunks (total_tokens/chunk_size): ~{total_tokens / self.chunk_size:.1f}")
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks (backward compatibility method).
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text or not text.strip():
            return []
        
        # Create a Document object for the text splitter
        doc = Document(page_content=text, metadata=metadata or {})
        
        # Use chunk_documents method
        return self.chunk_documents([doc])

