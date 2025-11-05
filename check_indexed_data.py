"""
Script to check chunks and embedding vectors indexed in Elasticsearch.
"""
import sys
from es_indexer import ESIndexer
from elasticsearch import Elasticsearch
import config
import json


def check_indexed_data(num_docs: int = 5):
    """
    Check chunks and embedding vectors from Elasticsearch.
    
    Args:
        num_docs: Number of documents to retrieve and display
    """
    print("="*80)
    print("Checking Indexed Data in Elasticsearch")
    print("="*80)
    
    # Initialize Elasticsearch client
    indexer = ESIndexer()
    client = indexer.client
    index_name = config.ELASTICSEARCH_INDEX_NAME
    
    # Get total document count
    try:
        stats = client.count(index=index_name)
        total_docs = stats['count']
        print(f"\nTotal documents in index: {total_docs}")
    except Exception as e:
        print(f"Error getting document count: {e}")
        return
    
    # Retrieve sample documents
    print(f"\nRetrieving {min(num_docs, total_docs)} sample documents...")
    print("="*80)
    
    try:
        # Search for documents
        response = client.search(
            index=index_name,
            body={
                "size": num_docs,
                "query": {"match_all": {}},
                "_source": ["text", "chunk_id", "metadata", "embedding"]
            }
        )
        
        hits = response['hits']['hits']
        
        for i, hit in enumerate(hits, 1):
            print(f"\n{'='*80}")
            print(f"Document {i}/{len(hits)}")
            print(f"{'='*80}")
            
            source = hit['_source']
            
            # Display chunk information
            chunk_id = source.get('chunk_id', 'N/A')
            print(f"\nChunk ID: {chunk_id}")
            
            # Display metadata
            metadata = source.get('metadata', {})
            if metadata:
                print(f"\nMetadata:")
                print(f"  File Name: {metadata.get('file_name', 'N/A')}")
                print(f"  Page: {metadata.get('page', 'N/A')}")
                if 'title' in metadata:
                    print(f"  Title: {metadata.get('title', 'N/A')}")
            
            # Display text (first 500 characters)
            text = source.get('text', '')
            print(f"\nText (first 500 chars):")
            print(f"  {text[:500]}...")
            print(f"\n  Text length: {len(text)} characters")
            
            # Display embedding vector information
            embedding = source.get('embedding', [])
            if embedding:
                print(f"\nEmbedding Vector:")
                print(f"  Dimension: {len(embedding)}")
                print(f"  First 10 values: {embedding[:10]}")
                print(f"  Last 10 values: {embedding[-10:]}")
                print(f"  Min value: {min(embedding):.6f}")
                print(f"  Max value: {max(embedding):.6f}")
                print(f"  Mean value: {sum(embedding)/len(embedding):.6f}")
                
                # Check if it's a zero vector
                if all(v == 0.0 for v in embedding):
                    print(f"  ⚠️  WARNING: This is a zero vector!")
                else:
                    print(f"  ✓ Vector contains non-zero values")
            else:
                print(f"\n⚠️  No embedding found for this document")
            
            print()
        
        # Statistics
        print(f"\n{'='*80}")
        print("Summary Statistics")
        print(f"{'='*80}")
        
        # Get all chunks to calculate statistics
        all_response = client.search(
            index=index_name,
            body={
                "size": 10000,  # Get all documents
                "query": {"match_all": {}},
                "_source": ["text", "embedding"]
            }
        )
        
        all_hits = all_response['hits']['hits']
        
        if all_hits:
            text_lengths = [len(hit['_source'].get('text', '')) for hit in all_hits]
            embedding_dims = [len(hit['_source'].get('embedding', [])) for hit in all_hits]
            
            print(f"\nText Statistics:")
            print(f"  Total chunks: {len(all_hits)}")
            print(f"  Avg text length: {sum(text_lengths)/len(text_lengths):.1f} characters")
            print(f"  Min text length: {min(text_lengths)} characters")
            print(f"  Max text length: {max(text_lengths)} characters")
            
            print(f"\nEmbedding Statistics:")
            print(f"  Embedding dimension: {embedding_dims[0] if embedding_dims else 'N/A'}")
            
            # Check for zero vectors
            zero_vectors = 0
            for hit in all_hits:
                emb = hit['_source'].get('embedding', [])
                if emb and all(v == 0.0 for v in emb):
                    zero_vectors += 1
            
            if zero_vectors > 0:
                print(f"  ⚠️  WARNING: {zero_vectors} documents have zero vectors!")
            else:
                print(f"  ✓ All documents have valid embedding vectors")
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        import traceback
        traceback.print_exc()


def check_specific_chunk(chunk_id: int):
    """Check a specific chunk by chunk_id."""
    indexer = ESIndexer()
    client = indexer.client
    index_name = config.ELASTICSEARCH_INDEX_NAME
    
    try:
        response = client.search(
            index=index_name,
            body={
                "query": {
                    "term": {"chunk_id": chunk_id}
                },
                "_source": ["text", "chunk_id", "metadata", "embedding"]
            }
        )
        
        hits = response['hits']['hits']
        if hits:
            hit = hits[0]
            source = hit['_source']
            
            print(f"\nChunk ID: {source.get('chunk_id')}")
            print(f"Text: {source.get('text', '')[:500]}...")
            print(f"Embedding dimension: {len(source.get('embedding', []))}")
            print(f"Embedding (first 20): {source.get('embedding', [])[:20]}")
        else:
            print(f"Chunk ID {chunk_id} not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check indexed chunks and embeddings")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=5,
        help="Number of documents to display (default: 5)"
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Check a specific chunk by ID"
    )
    
    args = parser.parse_args()
    
    if args.chunk_id is not None:
        check_specific_chunk(args.chunk_id)
    else:
        check_indexed_data(args.num_docs)

