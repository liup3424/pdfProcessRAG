# How to Access Elasticsearch and Check Embeddings

## Where Embeddings Are Stored

**Location:** Elasticsearch index `pdf_rag_index`

**Host:** `http://localhost:9200` (HTTP, not HTTPS - check your .env file)

**Data Structure:**
Each document in the index contains:
- `text`: The chunk text content
- `embedding`: Vector array (1024 dimensions) - **this is where your embeddings are stored**
- `chunk_id`: Unique chunk identifier
- `metadata`: File information (file_name, page_number, etc.)

## Access Methods

### 1. Using Python Script

```bash
# Check sample documents with embeddings
python check_elasticsearch.py --num-docs 5

# Show REST API examples
python check_elasticsearch.py --show-api
```

### 2. Using curl (Command Line)

**Get index information:**
```bash
curl -u elastic:YOUR_PASSWORD http://localhost:9200/pdf_rag_index
```

**Get document count:**
```bash
curl -u elastic:YOUR_PASSWORD http://localhost:9200/pdf_rag_index/_count
```

**Search all documents:**
```bash
curl -u elastic:YOUR_PASSWORD http://localhost:9200/pdf_rag_index/_search?pretty
```

**Get a specific document with embedding:**
```bash
curl -u elastic:YOUR_PASSWORD -X POST http://localhost:9200/pdf_rag_index/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 1,
    "_source": ["text", "embedding", "chunk_id", "metadata"]
  }'
```

**Get document by chunk_id:**
```bash
curl -u elastic:YOUR_PASSWORD -X POST http://localhost:9200/pdf_rag_index/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "term": {"chunk_id": 0}
    },
    "_source": ["text", "embedding", "chunk_id"]
  }'
```

### 3. Using Browser (if Elasticsearch REST API is accessible)

**Note:** By default, Elasticsearch doesn't allow direct browser access. You need to use:
- curl (command line)
- Python script
- Kibana Dev Tools (if installed)
- Postman or similar API client

**URL Format:**
```
http://localhost:9200/pdf_rag_index/_search?pretty
```

### 4. Using Python Code

```python
from es_indexer import ESIndexer

indexer = ESIndexer()
client = indexer.client

# Get a document
response = client.get(
    index="pdf_rag_index",
    id="DOCUMENT_ID"
)

# Get embedding vector
embedding = response['_source']['embedding']
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
```

## Important Notes

1. **HTTP vs HTTPS**: Your `.env` shows `http://localhost:9200` (HTTP), not HTTPS
   - Use `http://` not `https://` when accessing via browser/curl
   
2. **Authentication**: You need credentials from your `.env` file:
   - Username: `elastic` (from ELASTICSEARCH_USER)
   - Password: Your password (from ELASTICSEARCH_PASSWORD)
   - Or API Key (from ELASTICSEARCH_API_KEY)

3. **Embedding Vector Location**: 
   - Field name: `embedding`
   - Type: `dense_vector`
   - Dimension: 1024
   - Stored in: `_source.embedding` of each document

4. **Index Name**: `pdf_rag_index` (configurable in config.py)

## Quick Test

Run this to see your data:
```bash
python check_elasticsearch.py --num-docs 3
```

