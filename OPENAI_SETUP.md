# OpenAI API Setup Guide

## Quick Setup

### 1. Add to your `.env` file:

```bash
# LLM Configuration
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
```

### 2. Get Your OpenAI API Key:

1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-`)
6. Paste it in your `.env` file as `LLM_API_KEY`

### 3. Choose Your Model:

**Available OpenAI Models:**
- `gpt-4o` - Latest and most capable
- `gpt-4o-mini` - Faster and cheaper
- `gpt-4-turbo` - Previous generation
- `gpt-3.5-turbo` - Fastest and cheapest

**Note:** `gpt-5-nano` doesn't exist yet. Use one of the models above.

**For your `.env`:**
```bash
LLM_MODEL=gpt-4o-mini  # Recommended: fast and capable
# OR
LLM_MODEL=gpt-4o       # Most capable but more expensive
# OR
LLM_MODEL=gpt-3.5-turbo  # Cheapest option
```

## Complete .env Example

```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_API_KEY=your_api_key

# Embedding API
EMBEDDING_URL=http://test.2brain.cn:9800/v1/emb

# Re-ranking API
RERANK_URL=http://test.2brain.cn:2260/rerank

# LLM Configuration (OpenAI)
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```

## Testing

After setting up, test with:

```bash
python main.py query --query "What is relative value?"
```

The system will now use OpenAI to generate answers instead of simple templates.

## Cost Considerations

- **gpt-3.5-turbo**: ~$0.0015 per 1K tokens (cheapest)
- **gpt-4o-mini**: ~$0.15 per 1M input tokens (good balance)
- **gpt-4o**: ~$5 per 1M input tokens (most capable)

Each query uses approximately 500-2000 tokens depending on context length.

## Troubleshooting

1. **401 Unauthorized**: Check your API key is correct
2. **429 Rate Limit**: You've exceeded your quota, wait or upgrade plan
3. **Model not found**: Check the model name is correct (see OpenAI docs)

