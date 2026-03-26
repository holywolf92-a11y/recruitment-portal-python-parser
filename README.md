# Python CV Parser Service

FastAPI-based CV parsing service using OpenAI models for structured data extraction.

## Features

- ✅ HMAC authentication for secure requests
- ✅ OpenAI text and vision integration
- ✅ Structured JSON output
- ✅ Health check endpoint
- ✅ File upload support

## Environment Variables

```bash
PYTHON_HMAC_SECRET=your-secret-key  # Must match backend
OPENAI_API_KEY=sk-proj-xxxxx
PORT=8000
ENVIRONMENT=production
CV_VISION_MODEL=gpt-4o-mini
DOCUMENT_VISION_MODEL=gpt-4o-mini
```

Production currently pins both vision paths to `gpt-4o-mini` explicitly in Railway.

## API Endpoints

### GET /health
Health check endpoint

### POST /parse
Parse CV content with HMAC authentication

Headers:
- `X-HMAC-Signature`: HMAC signature of request body

Body:
```json
{
  "file_content": "CV text content",
  "file_name": "resume.pdf",
  "mime_type": "application/pdf"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "full_name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "skills": ["Python", "JavaScript"],
    "experience": [...],
    "education": [...]
  }
}
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Railway Deployment

1. Create new service from GitHub repo
2. Set environment variables in Railway dashboard
3. For production clarity, pin `CV_VISION_MODEL` and `DOCUMENT_VISION_MODEL` explicitly even though the service defaults to `gpt-4o-mini`
4. Deploy automatically

Railway will detect Python and use the Procfile.
