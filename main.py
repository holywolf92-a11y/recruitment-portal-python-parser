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
import fitz  # PyMuPDF
from PIL import Image
import base64
from supabase import create_client
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
    """Extract text from PDF bytes using PyMuPDF (fitz) first, with PyPDF2 fallback"""
    try:
        # Try PyMuPDF FIRST - it's much better at text extraction
        try:
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            pymupdf_text_parts = []
            for i, page in enumerate(pdf_doc):
                page_text = page.get_text()
                pymupdf_text_parts.append(page_text)
                logger.info(f"[PDFExtraction] PyMuPDF - Page {i+1}: Extracted {len(page_text)} characters")
            
            pymupdf_text = "\n".join(pymupdf_text_parts)
            logger.info(f"[PDFExtraction] PyMuPDF - Total extracted {len(pymupdf_text)} characters from {len(pdf_doc)} page(s)")
            
            # Log full text preview for debugging
            if pymupdf_text:
                preview = pymupdf_text[:1000].replace('\n', ' | ').replace('\r', ' ')
                logger.info(f"[PDFExtraction] PyMuPDF Text preview (first 1000 chars): {preview}")
            
            pdf_doc.close()
            
            # If PyMuPDF extracted good text, use it
            if len(pymupdf_text) > 100:
                logger.info(f"[PDFExtraction] Using PyMuPDF result ({len(pymupdf_text)} chars)")
                return pymupdf_text
            else:
                logger.warning(f"[PDFExtraction] PyMuPDF extracted only {len(pymupdf_text)} characters. Trying PyPDF2 as fallback...")
        except ImportError:
            logger.warning(f"[PDFExtraction] PyMuPDF (fitz) not available. Using PyPDF2.")
        except Exception as pymupdf_error:
            logger.warning(f"[PDFExtraction] PyMuPDF extraction failed: {pymupdf_error}. Trying PyPDF2 as fallback...")
        
        # Fallback to PyPDF2
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_parts.append(page_text)
            logger.info(f"[PDFExtraction] PyPDF2 - Page {i+1}: Extracted {len(page_text)} characters")

        full_text = "\n".join(text_parts)
        logger.info(f"[PDFExtraction] PyPDF2 - Total extracted {len(full_text)} characters from {len(pdf_reader.pages)} page(s)")
        
        # Log full text preview for debugging
        if full_text:
            preview = full_text[:1000].replace('\n', ' | ').replace('\r', ' ')
            logger.info(f"[PDFExtraction] PyPDF2 Text preview (first 1000 chars): {preview}")
        else:
            logger.warning(f"[PDFExtraction] WARNING: No text extracted from PDF! PDF might be image-based or corrupted.")
        
        return full_text
    except Exception as e:
        logger.error(f"[PDFExtraction] PDF extraction error: {e}")
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

async def categorize_document_with_ai(file_content: str, file_name: str, mime_type: str) -> dict:
    """
    Use OpenAI to categorize document and extract identity fields.
    
    Returns:
    {
        "category": "cv_resume" | "passport" | "certificates" | "contracts" | "medical_reports" | "photos" | "other_documents",
        "confidence": 0.0-1.0,
        "ocr_confidence": 0.0-1.0,
        "extracted_identity": {
            "name": "string or null",
            "father_name": "string or null",
            "cnic": "string or null",
            "passport_no": "string or null",
            "email": "string or null",
            "phone": "string or null",
            "date_of_birth": "string or null",
            "document_number": "string or null"
        }
    }
    """
    try:
        # Decode base64 content
        import base64
        # Ensure base64 string is properly padded (must be multiple of 4)
        if isinstance(file_content, str):
            file_content = file_content.strip()
            # Add padding if needed (base64 strings must be multiples of 4)
            missing_padding = len(file_content) % 4
            if missing_padding:
                file_content += '=' * (4 - missing_padding)
        
        try:
            # Use validate=False to be more lenient with padding
            file_bytes = base64.b64decode(file_content, validate=False)
        except Exception as decode_error:
            logger.error(f"[DocumentCategorization] Base64 decode error: {decode_error}, content length: {len(file_content) if file_content else 0}, first 50 chars: {file_content[:50] if file_content else 'None'}")
            raise HTTPException(status_code=400, detail=f"Invalid base64-encoded file content: {str(decode_error)}")
        
        # Extract text from document
        text_content = ""
        if mime_type == "application/pdf":
            text_content = extract_text_from_pdf(file_bytes)
        else:
            # For non-PDF files, try to decode as text
            try:
                text_content = file_bytes.decode('utf-8', errors='ignore')
            except:
                text_content = str(file_bytes)
        
        logger.info(f"[DocumentCategorization] Extracted {len(text_content)} characters from {file_name}")
        
        # Prepare OpenAI prompt for document categorization
        prompt = f"""
You are a document classification and identity extraction AI. Analyze the following document content and provide:

1. Document category (choose ONE):
   - cv_resume: CV, resume, curriculum vitae
   - passport: Passport copy, passport scan
   - certificates: Educational certificates, degrees, diplomas, training certificates
   - contracts: Employment contracts, offer letters, agreements
   - medical_reports: Medical test reports, health certificates, fitness certificates
   - photos: Passport photos, ID photos
   - other_documents: Any other document type

2. Confidence score (0.0 to 1.0) for the category classification

3. Extract identity fields from the document:
   - name: Full name of the person
   - father_name: Father's name (common in Pakistani documents)
   - cnic: Pakistani CNIC number (format: 12345-1234567-1 or 13 digits)
   - passport_no: Passport number (e.g., PA1234567, AB1234567)
   - email: Email address
   - phone: Phone number
   - date_of_birth: Date of birth (format: DD-MM-YYYY or YYYY-MM-DD)
   - document_number: Any other ID number found in the document
   - nationality: Nationality (e.g., Pakistani, Indian, etc.)
   - passport_expiry: Passport expiry date (format: DD-MM-YYYY or YYYY-MM-DD)
   - expiry_date: Alternative field for passport expiry date
   - issue_date: Passport issue date (format: DD-MM-YYYY or YYYY-MM-DD)
   - place_of_issue: Place where passport was issued (e.g., Islamabad, Karachi)

Return ONLY valid JSON with this exact structure:
{{
  "category": "category_name",
  "confidence": 0.95,
  "ocr_confidence": 0.90,
  "extracted_identity": {{
    "name": "string or null",
    "father_name": "string or null",
    "cnic": "string or null",
    "passport_no": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "date_of_birth": "string or null",
    "document_number": "string or null",
    "nationality": "string or null (e.g., Pakistani, Indian)",
    "passport_expiry": "string or null (format: DD-MM-YYYY or YYYY-MM-DD)",
    "expiry_date": "string or null (alternative field for passport expiry)",
    "issue_date": "string or null (format: DD-MM-YYYY or YYYY-MM-DD)",
    "place_of_issue": "string or null (e.g., Islamabad, Karachi)"
  }}
}}

Document filename: {file_name}
Document content (full text extracted from PDF):
{text_content}

IMPORTANT: Extract ALL identity fields from the document content above. Look for:
- Full Name / Name
- Passport Number / Passport No
- Nationality / Country
- Date of Birth / DOB
- Issue Date
- Expiry Date / Expiry
- Place of Issue

The document content is provided above - extract the information even if the format is slightly different.
"""

        # Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document classification expert. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        parsed_result = json.loads(result_text)
        
        # Ensure extracted_identity exists and filter out null values for logging
        extracted_identity = parsed_result.get('extracted_identity', {})
        
        # Ensure all passport fields are present (even if null)
        if not extracted_identity:
            extracted_identity = {}
        
        # Ensure backward compatibility: if dob exists but date_of_birth doesn't, copy it
        if extracted_identity.get("dob") and not extracted_identity.get("date_of_birth"):
            extracted_identity["date_of_birth"] = extracted_identity["dob"]
        
        # Ensure passport_expiry or expiry_date is set (prefer passport_expiry)
        if extracted_identity.get("expiry_date") and not extracted_identity.get("passport_expiry"):
            extracted_identity["passport_expiry"] = extracted_identity["expiry_date"]
        
        parsed_result['extracted_identity'] = extracted_identity
        
        non_null_identity = {k: v for k, v in extracted_identity.items() if v is not None} if extracted_identity else {}
        
        logger.info(f"[DocumentCategorization] Categorized as: {parsed_result.get('category')} (confidence: {parsed_result.get('confidence')})")
        if non_null_identity:
            logger.info(f"[DocumentCategorization] Extracted identity fields: {list(non_null_identity.keys())}")
        else:
            logger.warning(f"[DocumentCategorization] No identity fields extracted from document")
        
        return parsed_result
        
    except json.JSONDecodeError as e:
        logger.error(f"[DocumentCategorization] JSON decode error: {e}, response: {result_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        logger.error(f"[DocumentCategorization] Categorization error: {e}")
        raise HTTPException(status_code=500, detail=f"Document categorization failed: {str(e)}")

@app.post("/categorize-document")
async def categorize_document(
    request: Request,
    x_hmac_signature: str = Header(None)
):
    """
    Categorize document and extract identity fields using AI.
    Protected with HMAC authentication.
    """
    
    # Verify HMAC signature
    if not x_hmac_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")
    
    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    
    try:
        # Parse request body
        payload = json.loads(body)
        file_content = payload.get('file_content')
        file_name = payload.get('file_name', 'unknown')
        mime_type = payload.get('mime_type', 'application/pdf')
        
        # Log what we received
        logger.info(f"[CategorizeDocument] Received request - fileName: {file_name}, mimeType: {mime_type}")
        logger.info(f"[CategorizeDocument] file_content type: {type(file_content)}, length: {len(file_content) if file_content else 0}")
        if file_content:
            logger.info(f"[CategorizeDocument] file_content first 100 chars: {file_content[:100]}")
            logger.info(f"[CategorizeDocument] file_content last 20 chars: {file_content[-20:]}")
        
        if not file_content:
            raise HTTPException(status_code=400, detail="file_content is required")
        
        # Validate base64 string
        if isinstance(file_content, str):
            original_content = file_content
            # Remove any whitespace and newlines
            file_content = file_content.strip().replace('\n', '').replace('\r', '')
            
            # Log after cleaning
            if original_content != file_content:
                logger.info(f"[CategorizeDocument] Cleaned whitespace - original length: {len(original_content)}, cleaned length: {len(file_content)}")
            
            # Check if this looks like base64 (only base64 chars: A-Z, a-z, 0-9, +, /, =)
            import re
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(file_content):
                # This is not base64 - it's probably raw text content
                logger.error(f"[CategorizeDocument] Received non-base64 content. First 100 chars: {file_content[:100]}")
                logger.error(f"[CategorizeDocument] Content contains invalid chars. Sample: {repr(file_content[:200])}")
                logger.error(f"[CategorizeDocument] Original content (before cleaning): {repr(original_content[:200])}")
                raise HTTPException(status_code=400, detail="file_content must be base64-encoded, not raw text")
            
            # Base64 strings should be multiples of 4 (with padding)
            # Add padding if needed
            missing_padding = len(file_content) % 4
            if missing_padding:
                file_content += '=' * (4 - missing_padding)
            
            # Log validated base64 before passing to categorize function
            logger.info(f"[CategorizeDocument] Validated base64 - length: {len(file_content)}, first 50 chars: {file_content[:50]}")
        
        # Categorize document
        result = await categorize_document_with_ai(file_content, file_name, mime_type)
        # Ensure all required fields are present, even if null
        return {
            "success": True,
            "category": result.get("category", None),
            "confidence": result.get("confidence", None),
            "ocr_confidence": result.get("ocr_confidence", None),
            "extracted_identity": result.get("extracted_identity", None),
            "mismatch_fields": result.get("mismatch_fields", []),
            "raw_text": None  # Don't return raw text to reduce payload size
        }
    except Exception as e:
        logger.error(f"[CategorizeDocument] Error: {e}")
        # Always return all required fields as null/empty on error
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "ocr_confidence": None,
            "extracted_identity": None,
            "mismatch_fields": [],
            "raw_text": None,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
