"""
PDF Processing Module
Extracts text from PDF files using LangChain's PyMuPDFLoader.
"""
import os
from typing import Dict, List
from langchain_community.document_loaders import PyMuPDFLoader

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document


class PDFProcessor:
    """Process PDF files and extract text content using PyMuPDFLoader."""
    
    def __init__(self):
        """Initialize PDF processor."""
        pass
    
    def load_pages(self, pdf_path: str) -> List[Document]:
        """
        Load PDF pages using PyMuPDFLoader.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        # Add file metadata to each page
        file_name = os.path.basename(pdf_path)
        for page in pages:
            if page.metadata:
                page.metadata["file_path"] = pdf_path
                page.metadata["file_name"] = file_name
        
        return pages
    
    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file using PyMuPDFLoader (backward compatibility).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: PDF metadata (title, author, pages, etc.)
                - pages: List of text per page
        """
        pages = self.load_pages(pdf_path)
        
        # Extract text from all documents
        text_content = []
        pages_text = []
        
        # Get metadata from first document
        first_doc_metadata = pages[0].metadata if pages else {}
        
        # Combine all page content
        for doc in pages:
            page_text = doc.page_content
            if page_text:
                page_num = doc.metadata.get("page", len(pages_text) + 1)
                pages_text.append({
                    "page_number": page_num,
                    "text": page_text
                })
                text_content.append(page_text)
        
        # Build metadata dictionary
        metadata = {
            "title": first_doc_metadata.get("title", ""),
            "author": first_doc_metadata.get("author", ""),
            "subject": first_doc_metadata.get("subject", ""),
            "creator": first_doc_metadata.get("creator", ""),
            "producer": first_doc_metadata.get("producer", ""),
            "creation_date": str(first_doc_metadata.get("creation_date", "")),
            "modification_date": str(first_doc_metadata.get("mod_date", "")),
            "total_pages": len(pages),
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path)
        }
        
        # Combine all text
        full_text = "\n\n".join(text_content)
        
        return {
            "text": full_text,
            "metadata": metadata,
            "pages": pages_text
        }

