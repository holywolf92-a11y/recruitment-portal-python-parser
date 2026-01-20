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
import httpx
import PyPDF2
import io
import fitz  # PyMuPDF
from PIL import Image
import base64
from supabase import create_client

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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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
  "nationality": "string or null (country of origin/citizenship)",
  "position": "string or null (desired job position/profession/title)",
  "experience_years": "number or null (total years of professional work experience)",
  "country_of_interest": "string or null (country they want to work in, check objective/career goals)",
  "linkedin_url": "string or null",
  "summary": "string or null",
  "professional_summary": "string or null (2-3 sentence career summary)",
  "skills": ["array of strings - all technical and professional skills"],
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
  "languages": ["array of strings"],
  "previous_employment": "string or null (brief summary of work history)",
  "passport_expiry": "string or null (format: YYYY-MM-DD)"
}}

IMPORTANT Guidelines:
- Extract nationality from personal info, birthplace, or passport details
- Extract position from objective, desired role, or most recent job title
- Calculate experience_years from work history timeline if not stated
- Look for country_of_interest in objective/goal statements (e.g., "seeking opportunities in UAE")
- Extract ALL skills mentioned (technical, soft skills, software, languages, certifications)
- For skills, include programming languages, tools, frameworks, soft skills
- Be thorough in skills extraction - don't miss any mentioned abilities

CV Content:
{content[:4000]}

Return only the JSON object, no explanation.
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model: ~$0.15/1M input tokens (200x cheaper than GPT-4)
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
        
        # Add "missing" default for country_of_interest if null or empty
        if not parsed_data.get('country_of_interest'):
            parsed_data['country_of_interest'] = 'missing'
        
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
        
        # Extract profile photo (non-blocking - won't fail if extraction fails)
        profile_photo_url = extract_profile_photo_from_pdf(file_content, attachment_id or "unknown")
        
        # Parse with OpenAI
        parsed_data = await parse_cv_with_openai(text_content, attachment_id or "unknown")
        
        # Add profile photo URL to parsed data if extracted
        if profile_photo_url:
            parsed_data['profile_photo_url'] = profile_photo_url
        
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

def extract_profile_photo_from_pdf(pdf_content: bytes, attachment_id: str) -> Optional[str]:
    """
    Extract profile photo from PDF and upload to Supabase Storage.
    Returns the public URL of the uploaded photo, or None if no photo found.
    """
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        if pdf_document.page_count == 0:
            logger.info(f"[PhotoExtraction] PDF has no pages for attachment {attachment_id}")
            return None
        
        logger.info(f"[PhotoExtraction] PDF has {pdf_document.page_count} pages")
        
        # Check first page only (profile photos are usually on page 1)
        first_page = pdf_document[0]
        image_list = first_page.get_images(full=True)
        
        logger.info(f"[PhotoExtraction] Found {len(image_list)} images on first page")
        
        if not image_list:
            logger.info(f"[PhotoExtraction] No images found in PDF first page for {attachment_id}")
            return None
        
        # Find largest image (likely the profile photo)
        largest_image = None
        largest_size = 0
        all_sizes = []
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Calculate image size
            image_size = len(image_bytes)
            all_sizes.append((image_size, image_ext))
            
            logger.info(f"[PhotoExtraction] Image {img_index}: {image_size} bytes ({image_ext})")
            
            # Profile photos are typically 5KB-1MB (relaxed from 10KB-500KB)
            # Skip very small images (icons/logos) and very large images (full-page scans)
            if 5000 < image_size < 1000000 and image_size > largest_size:
                largest_image = {
                    "bytes": image_bytes,
                    "ext": image_ext,
                    "size": image_size
                }
                largest_size = image_size
                logger.info(f"[PhotoExtraction] Selected image {img_index} as potential profile photo ({image_size} bytes)")
        
        pdf_document.close()
        
        if not largest_image:
            logger.info(f"[PhotoExtraction] No suitable profile photo found. Images found: {all_sizes}")
            return None
        
        logger.info(f"[PhotoExtraction] Uploading profile photo ({largest_image['size']} bytes)")
        
        # Upload to Supabase Storage
        photo_url = upload_photo_to_supabase(
            largest_image["bytes"],
            attachment_id,
            largest_image["ext"]
        )
        
        if photo_url:
            logger.info(f"[PhotoExtraction] Successfully extracted and uploaded profile photo: {photo_url}")
        else:
            logger.warning(f"[PhotoExtraction] Photo extracted but upload returned None")
        
        return photo_url
        
    except Exception as e:
        logger.warning(f"[PhotoExtraction] Photo extraction failed (non-critical): {e}")
        import traceback
        logger.warning(f"[PhotoExtraction] Traceback: {traceback.format_exc()}")
        return None  # Graceful fallback - don't fail CV parsing if photo extraction fails

def upload_photo_to_supabase(image_bytes: bytes, attachment_id: str, file_ext: str) -> str:
    """Upload extracted photo to Supabase Storage bucket using supabase-py client"""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            logger.warning("[PhotoUpload] Supabase credentials not configured, skipping photo upload")
            return None
        
        logger.info(f"[PhotoUpload] Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        # Bucket and file path
        bucket_name = "documents"
        file_path = f"candidate_photos/{attachment_id}/profile.{file_ext}"
        
        logger.info(f"[PhotoUpload] Uploading to: {bucket_name}/{file_path}")
        logger.info(f"[PhotoUpload] Image size: {len(image_bytes):,} bytes")
        
        # Upload using Supabase client
        response = supabase.storage.from_(bucket_name).upload(file_path, image_bytes)
        
        logger.info(f"[PhotoUpload] Upload response: {response}")
        
        # Return public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_path}"
        logger.info(f"[PhotoUpload] Success! Public URL: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"[PhotoUpload] Photo upload error: {e}")
        import traceback
        logger.error(f"[PhotoUpload] Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
