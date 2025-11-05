"""
Script to check chunks and embedding vectors from Elasticsearch.
Shows how embeddings are stored and how to access them.
"""
from es_indexer import ESIndexer
import config
import json


def show_embedding_storage_info():
    """Show where embeddings are stored and how to access them."""
    print("="*80)
    print("Embedding Vector Storage Information")
    print("="*80)
    
    print(f"\n📍 Storage Location:")
    print(f"  Index Name: {config.ELASTICSEARCH_INDEX_NAME}")
    print(f"  Elasticsearch Host: {config.ELASTICSEARCH_HOST}")
    print(f"  Index: {config.ELASTICSEARCH_INDEX_NAME}")
    
    print(f"\n📊 Data Structure:")
    print(f"  Each document contains:")
    print(f"    - text: The chunk text")
    print(f"    - embedding: Vector array (dimension: {config.EMBEDDING_DIMENSION})")
    print(f"    - chunk_id: Unique chunk identifier")
    print(f"    - metadata: File information (file_name, page_number, etc.)")
    
    print(f"\n🔍 Access Methods:")
    print(f"  1. Python API (this script)")
    print(f"  2. Elasticsearch REST API: {config.ELASTICSEARCH_HOST}/{config.ELASTICSEARCH_INDEX_NAME}/_search")
    print(f"  3. Kibana Dev Tools (if available)")
    
    print("\n" + "="*80)


def check_sample_documents(num_docs: int = 3):
    """Check sample documents with their embeddings."""
    print(f"\n📋 Checking {num_docs} Sample Documents")
    print("="*80)
    
    indexer = ESIndexer()
    client = indexer.client
    index_name = config.ELASTICSEARCH_INDEX_NAME
    
    try:
        # Get total count
        stats = client.count(index=index_name)
        total_docs = stats['count']
        print(f"Total documents in index: {total_docs}\n")
        
        # Retrieve sample documents
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
            print(f"\n{'─'*80}")
            print(f"Document {i}/{len(hits)} (ID: {hit['_id']})")
            print(f"{'─'*80}")
            
            source = hit['_source']
            
            # Chunk info
            chunk_id = source.get('chunk_id', 'N/A')
            print(f"\n📄 Chunk ID: {chunk_id}")
            
            # Metadata
            metadata = source.get('metadata', {})
            if metadata:
                print(f"\n📝 Metadata:")
                print(f"   File: {metadata.get('file_name', 'N/A')}")
                if 'page' in metadata:
                    print(f"   Page: {metadata.get('page')}")
            
            # Text
            text = source.get('text', '')
            print(f"\n📝 Text Preview (first 200 chars):")
            print(f"   {text[:200]}...")
            print(f"   Total length: {len(text)} characters")
            
            # Embedding
            embedding = source.get('embedding', [])
            if embedding:
                print(f"\n🔢 Embedding Vector:")
                print(f"   Dimension: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
                print(f"   Last 5 values: {embedding[-5:]}")
                print(f"   Min: {min(embedding):.6f}, Max: {max(embedding):.6f}, Mean: {sum(embedding)/len(embedding):.6f}")
                
                # Check if zero vector
                if all(v == 0.0 for v in embedding):
                    print(f"   ⚠️  WARNING: Zero vector detected!")
                else:
                    print(f"   ✓ Valid vector with non-zero values")
            else:
                print(f"\n⚠️  No embedding found")
            
            print()
        
        # Summary
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Total indexed documents: {total_docs}")
        print(f"Embedding dimension: {config.EMBEDDING_DIMENSION}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def show_rest_api_examples():
    """Show REST API examples for accessing Elasticsearch."""
    print("\n" + "="*80)
    print("REST API Access Examples")
    print("="*80)
    
    host = config.ELASTICSEARCH_HOST
    index = config.ELASTICSEARCH_INDEX_NAME
    
    # Get credentials
    user = config.ELASTICSEARCH_USER
    password = config.ELASTICSEARCH_PASSWORD
    api_key = config.ELASTICSEARCH_API_KEY
    
    auth_str = ""
    if api_key:
        auth_str = f"-H 'Authorization: ApiKey {api_key}'"
    elif password:
        auth_str = f"-u {user}:{password}"
    
    print(f"\n1. Get index information:")
    print(f"   curl -k {auth_str} {host}/{index}")
    
    print(f"\n2. Get document count:")
    print(f"   curl -k {auth_str} {host}/{index}/_count")
    
    print(f"\n3. Search all documents (without embeddings):")
    print(f"   curl -k {auth_str} {host}/{index}/_search?pretty")
    
    print(f"\n4. Get a specific document by ID:")
    print(f"   curl -k {auth_str} {host}/{index}/_doc/DOCUMENT_ID?pretty")
    
    print(f"\n5. Search for a specific chunk_id:")
    print(f"   curl -k {auth_str} -X POST {host}/{index}/_search?pretty \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"query\": {{\"term\": {{\"chunk_id\": 0}}}}}}'")
    
    print(f"\n6. Get a document with embedding vector:")
    print(f"   curl -k {auth_str} -X POST {host}/{index}/_search?pretty \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"size\": 1, \"_source\": [\"text\", \"embedding\", \"chunk_id\"]}}'")
    
    print(f"\n7. Vector similarity search (nearest neighbors):")
    print(f"   curl -k {auth_str} -X POST {host}/{index}/_search?pretty \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{")
    print(f"          \"query\": {{")
    print(f"            \"script_score\": {{")
    print(f"              \"query\": {{\"match_all\": {{}}}}, ")
    print(f"              \"script\": {{")
    print(f"                \"source\": \"cosineSimilarity(params.query_vector, 'embedding') + 1.0\",")
    print(f"                \"params\": {{\"query_vector\": [0.1, 0.2, ...]}}")
    print(f"              }}")
    print(f"            }}")
    print(f"          }}")
    print(f"        }}'")
    
    print(f"\n💡 Note: Replace DOCUMENT_ID with an actual document ID from the index")
    print(f"💡 Note: For HTTPS, use -k flag to ignore SSL certificate errors")
    print(f"💡 Note: Use --user {user}:{password} if using basic auth")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Elasticsearch indexed data")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=3,
        help="Number of documents to display (default: 3)"
    )
    parser.add_argument(
        "--show-api",
        action="store_true",
        help="Show REST API examples"
    )
    
    args = parser.parse_args()
    
    show_embedding_storage_info()
    check_sample_documents(args.num_docs)
    
    if args.show_api:
        show_rest_api_examples()

