from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hmac
import hashlib
import os
import logging
from typing import Optional
import openai
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV Parser Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
HMAC_SECRET = os.getenv("PYTHON_HMAC_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

if not HMAC_SECRET:
    raise ValueError("PYTHON_HMAC_SECRET environment variable is required")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

class ParseRequest(BaseModel):
    file_content: str
    file_name: str
    mime_type: str

class ParseResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

def verify_hmac(signature: str, body: bytes) -> bool:
    """Verify HMAC signature"""
    try:
        expected_signature = hmac.new(
            HMAC_SECRET.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"HMAC verification error: {e}")
        return False

async def parse_cv_with_openai(content: str, filename: str) -> dict:
    """Parse CV content using OpenAI"""
    try:
        prompt = f"""
You are a CV/Resume parser. Extract structured information from the following CV content.
Return ONLY valid JSON with these exact fields (use null for missing data):

{{
  "full_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "linkedin_url": "string or null",
  "summary": "string or null",
  "skills": ["array of strings"],
  "experience": [
    {{
      "title": "string",
      "company": "string",
      "location": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string or null"
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "location": "string or null",
      "graduation_date": "string or null"
    }}
  ],
  "certifications": ["array of strings"],
  "languages": ["array of strings"]
}}

CV Content:
{content[:4000]}

Return only the JSON object, no explanation.
"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise CV parser that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        parsed_data = json.loads(result_text)
        
        logger.info(f"Successfully parsed CV: {filename}")
        return parsed_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, response: {result_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse OpenAI response as JSON: {str(e)}")
    except Exception as e:
        logger.error(f"OpenAI parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"CV parsing failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CV Parser",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_configured": bool(OPENAI_API_KEY),
        "hmac_configured": bool(HMAC_SECRET)
    }

@app.post("/parse", response_model=ParseResponse)
async def parse_cv(
    request: Request,
    parse_request: ParseRequest,
    x_hmac_signature: str = Header(None)
):
    """Parse CV with HMAC authentication"""
    
    # Verify HMAC signature
    if not x_hmac_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")
    
    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    
    try:
        # Parse the CV
        parsed_data = await parse_cv_with_openai(
            parse_request.file_content,
            parse_request.file_name
        )
        
        return ParseResponse(
            success=True,
            data=parsed_data
        )
    
    except Exception as e:
        logger.error(f"Parse error: {e}")
        return ParseResponse(
            success=False,
            error=str(e)
        )

@app.post("/parse-file")
async def parse_cv_file(
    request: Request,
    file: UploadFile = File(...),
    x_hmac_signature: str = Header(None)
):
    """Parse CV file directly with HMAC authentication"""
    
    # Verify HMAC signature
    if not x_hmac_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")
    
    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        
        # Parse the CV
        parsed_data = await parse_cv_with_openai(content_str, file.filename)
        
        return ParseResponse(
            success=True,
            data=parsed_data
        )
    
    except Exception as e:
        logger.error(f"File parse error: {e}")
        return ParseResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
