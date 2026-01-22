from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hmac
import hashlib
import os
import logging
from typing import Optional, Dict, Any
import openai
from datetime import datetime
import json
import httpx
import PyPDF2
import io
import base64
import re

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

class CategorizeRequest(BaseModel):
    file_content: str
    file_name: str
    mime_type: str
    candidate_data: Optional[Dict[str, Any]] = None

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

async def categorize_document_with_ai(content: str, filename: str, candidate_data: dict = None) -> dict:
    """Categorize document and extract identity fields using OpenAI"""
    try:
        prompt = f"""
You are a document classifier and identity extractor. Analyze the document and:
1. Classify it into ONE category
2. Extract identity information

CATEGORIES (choose ONE):
- cv_resume: CV, resume, or employment history
- passport: Passport document
- certificates: Degrees, diplomas, certifications
- contracts: Employment contracts, agreements
- medical_reports: Medical certificates, health reports
- photos: Profile pictures, headshots
- other_documents: Any other document type

IDENTITY FIELDS TO EXTRACT (return null if not found):
- name: Full name of the person
- father_name: Father's name (if available)
- cnic: Pakistani CNIC (format: 12345-1234567-1)
- passport_no: Passport number (e.g., PA1234567, AB1234567)
- email: Email address
- phone: Phone number
- date_of_birth: Date of birth (format: DD-MM-YYYY or YYYY-MM-DD)
- document_number: Any other ID number found
- nationality: Nationality (e.g., Pakistani, Indian, etc.)
- passport_expiry: Passport expiry date (format: DD-MM-YYYY or YYYY-MM-DD)
- expiry_date: Alternative field for passport expiry date
- issue_date: Passport issue date (format: DD-MM-YYYY or YYYY-MM-DD)
- place_of_issue: Place where passport was issued (e.g., Islamabad, Karachi)

Document Content:
{content[:4000]}

Candidate Info (for reference):
{json.dumps(candidate_data) if candidate_data else "None"}

Return ONLY valid JSON:
{{
  "category": "one_of_the_categories_above",
  "confidence": 0.0_to_1.0,
  "identity_fields": {{
    "name": "string or null",
    "father_name": "string or null",
    "cnic": "string or null",
    "passport_no": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "date_of_birth": "string or null (format: DD-MM-YYYY or YYYY-MM-DD)",
    "dob": "string or null (same as date_of_birth, for backward compatibility)",
    "document_number": "string or null",
    "nationality": "string or null (e.g., Pakistani, Indian)",
    "passport_expiry": "string or null (format: DD-MM-YYYY or YYYY-MM-DD)",
    "expiry_date": "string or null (alternative field for passport expiry)",
    "issue_date": "string or null (format: DD-MM-YYYY or YYYY-MM-DD)",
    "place_of_issue": "string or null (e.g., Islamabad, Karachi)"
  }}
}}
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise document classifier that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
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
        # Ensure required fields are present and correct type
        category = parsed_data.get('category') or 'other_documents'
        confidence = parsed_data.get('confidence')
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        identity_fields = parsed_data.get('identity_fields')
        if not isinstance(identity_fields, dict):
            identity_fields = {
                "name": None,
                "father_name": None,
                "cnic": None,
                "passport_no": None,
                "email": None,
                "phone": None,
                "date_of_birth": None,
                "dob": None,
                "document_number": None,
                "nationality": None,
                "passport_expiry": None,
                "expiry_date": None,
                "issue_date": None,
                "place_of_issue": None
            }
        
        # Ensure backward compatibility: if dob exists but date_of_birth doesn't, copy it
        if identity_fields.get("dob") and not identity_fields.get("date_of_birth"):
            identity_fields["date_of_birth"] = identity_fields["dob"]
        
        # Ensure passport_expiry or expiry_date is set (prefer passport_expiry)
        if identity_fields.get("expiry_date") and not identity_fields.get("passport_expiry"):
            identity_fields["passport_expiry"] = identity_fields["expiry_date"]
        logger.info(f"Categorized document: {filename} as {category} with confidence {confidence}")
        return {
            "category": category,
            "confidence": confidence,
            "identity_fields": identity_fields
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, response: {result_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse OpenAI response as JSON: {str(e)}")
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        raise HTTPException(status_code=500, detail=f"Document categorization failed: {str(e)}")

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
            model="gpt-4o-mini",
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

@app.post("/categorize-document")
async def categorize_document(
    request: Request,
    categorize_request: CategorizeRequest,
    x_hmac_signature: str = Header(None)
):
    """Categorize document and extract identity fields with HMAC authentication"""

    # Verify HMAC signature
    if not x_hmac_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")

    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    try:
        # Decode base64 file content
        try:
            file_bytes = base64.b64decode(categorize_request.file_content)
            file_text = file_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Failed to decode as text, trying PDF extraction: {e}")
            # If not text, try PDF extraction
            try:
                file_bytes = base64.b64decode(categorize_request.file_content)
                file_text = extract_text_from_pdf(file_bytes)
            except Exception as pdf_error:
                logger.error(f"Failed to extract text from file: {pdf_error}")
                raise HTTPException(status_code=400, detail=f"Could not extract text from file: {str(pdf_error)}")

        # Categorize the document
        result = await categorize_document_with_ai(
            file_text,
            categorize_request.file_name,
            categorize_request.candidate_data
        )
        # Always return required fields, even if missing
        return {
            "success": True,
            "category": result.get("category", "other_documents"),
            "confidence": float(result.get("confidence", 0.0)),
            "identity_fields": result.get("identity_fields") or {
                "name": None,
                "father_name": None,
                "cnic": None,
                "passport_no": None,
                "email": None,
                "phone": None,
                "date_of_birth": None,
                "dob": None,
                "document_number": None,
                "nationality": None,
                "passport_expiry": None,
                "expiry_date": None,
                "issue_date": None,
                "place_of_issue": None
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/parse-cv")
async def parse_cv_from_url(
    request: Request,
    x_signature: str = Header(None)
):
    """Parse CV from URL - backend worker endpoint"""

    # Verify HMAC signature
    if not x_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")

    body = await request.body()
    if not verify_hmac(x_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    try:
        # Parse request body
        payload = json.loads(body)
        file_url = payload.get('file_url')
        attachment_id = payload.get('attachment_id')
        file_hash = payload.get('file_hash')

        if not file_url:
            raise HTTPException(status_code=400, detail="file_url is required")

        # Fetch file from URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content

        # Extract text from PDF
        text_content = extract_text_from_pdf(file_content)

        # Parse with OpenAI
        parsed_data = await parse_cv_with_openai(text_content, attachment_id or "unknown")

        # Return in expected format
        return {
            "schema_version": "v1",
            "attachment_id": attachment_id,
            "file_hash": file_hash,
            "extracted_at": datetime.utcnow().isoformat(),
            "candidate": parsed_data
        }

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch file: {str(e)}")
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())

        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
