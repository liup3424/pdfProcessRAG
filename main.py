"""
Main CLI Entry Point
Command-line interface for the RAG system.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from src.setup_logging import setup_logging
from src import config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG System for PDF Processing")
    parser.add_argument(
        "command",
        choices=["process", "query", "test"],
        help="Command to execute: process (a PDF), query (the system), or test (connection)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file (for process command)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text (for query command)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of top results to return after re-ranking (default: {config.RERANK_TOP_K} from config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: logs/rag_system_TIMESTAMP.log)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    log_file = setup_logging(log_level=log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting RAG system - Command: {args.command}")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    if args.command == "test":
        print("Testing Elasticsearch connection...")
        if rag.test_connection():
            print("✓ Connection successful!")
        else:
            print("✗ Connection failed!")
            return 1
    
    elif args.command == "process":
        if not args.pdf:
            print("Error: --pdf is required for process command")
            return 1
        
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file not found: {args.pdf}")
            return 1
        
        success = rag.process_pdf(args.pdf)
        return 0 if success else 1
    
    elif args.command == "query":
        if not args.query:
            print("Error: --query is required for query command")
            return 1
        
        print(f"Querying: {args.query}")
        result = rag.query(
            args.query, 
            top_k_retrieval=config.RETRIEVAL_TOP_K,
            top_k_rerank=args.top_k or config.RERANK_TOP_K
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        print(f"SOURCES ({result['num_sources']}):")
        print("="*80)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source.get('file_name', 'Unknown')}")
            if source.get("page_number"):
                print(f"    Page: {source['page_number']}")
            print(f"    Text: {source['text']}...")
        
        return 0


if __name__ == "__main__":
    exit(main())