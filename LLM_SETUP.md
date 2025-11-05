# LLM API Setup Guide

## Overview

The LLM API is used for answer generation in the RAG system. It's **optional** - if not configured, the system will use a simple template-based answer generator.

## Configuration

### Option 1: Set in `.env` file (Recommended)

Add these lines to your `.env` file:

```bash
# LLM API Configuration
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-3.5-turbo
```

### Option 2: Set in `config.py` directly

You can also set defaults in `config.py` (not recommended for production).

## Supported LLM Providers

### 1. OpenAI

**API URL:** `https://api.openai.com/v1/chat/completions`

**How to get:**
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key

**Setup in `.env`:**
```bash
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-3.5-turbo
# Note: You'll need to add API key authentication in the code
```

**Models available:**
- `gpt-3.5-turbo` (cheaper, faster)
- `gpt-4` (more capable, expensive)
- `gpt-4-turbo-preview`

### 2. Anthropic Claude

**API URL:** `https://api.anthropic.com/v1/messages`

**How to get:**
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Go to API Keys
4. Create a new API key

**Setup in `.env`:**
```bash
LLM_API_URL=https://api.anthropic.com/v1/messages
LLM_MODEL=claude-3-sonnet-20240229
```

### 3. Local/Open Source Models (via API)

If you have a local LLM server (like Ollama, vLLM, etc.):

**Example with Ollama:**
```bash
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=llama2
```

**Example with local API server:**
```bash
LLM_API_URL=http://localhost:8000/v1/chat/completions
LLM_MODEL=your-model-name
```

### 4. Other Providers

- **Google Gemini:** `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent`
- **Azure OpenAI:** `https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT/chat/completions?api-version=2023-12-01-preview`
- **Cohere:** `https://api.cohere.ai/v1/generate`

## API Format Requirements

The current implementation expects OpenAI-compatible format:

**Request:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant..."
    },
    {
      "role": "user",
      "content": "Question based on context..."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response:**
```json
{
  "choices": [
    {
      "message": {
        "content": "Generated answer..."
      }
    }
  ]
}
```

## Authentication

Currently, the code doesn't include API key authentication. You'll need to:

1. **Add API key to request headers** (modify `answer_generator.py`)
2. Or use a proxy/gateway that handles authentication

### Example: Adding OpenAI API Key

Modify `answer_generator.py` to include authentication:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}
```

## Testing

1. **Test without LLM API** (current state):
   ```bash
   python main.py query --query "Your question"
   ```
   - Uses simple template-based answers

2. **Test with LLM API** (after setup):
   ```bash
   # Set LLM_API_URL in .env
   python main.py query --query "Your question"
   ```
   - Uses LLM for answer generation

## Current Status

- **Without LLM API:** System works but returns simple template-based answers
- **With LLM API:** System will generate more sophisticated answers using the LLM

## Troubleshooting

1. **API not responding:**
   - Check URL is correct
   - Verify network connectivity
   - Check if API requires authentication

2. **Authentication errors:**
   - Ensure API key is set correctly
   - Check API key permissions

3. **Format errors:**
   - Verify API response format matches expected format
   - Check API documentation for correct format

