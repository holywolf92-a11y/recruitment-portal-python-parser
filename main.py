import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger("python-parser")

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hmac
import hashlib
import os

from pathlib import Path
from dotenv import load_dotenv

# Load .env: parser dir first, then fallback to Recruitment Automation Portal (2) python-parser
_env_dir = Path(__file__).resolve().parent
_load = load_dotenv(_env_dir / ".env")
if not _load:
    _fallback = _env_dir.parent / "Recruitment Automation Portal (2)" / "python-parser" / ".env"
    if _fallback.exists():
        load_dotenv(_fallback)
from typing import Optional, Dict, Any
from collections import OrderedDict
import openai
from datetime import datetime

from split_and_categorize import run_split_and_categorize
import json
import httpx
import PyPDF2
import io
import fitz  # PyMuPDF
from PIL import Image
import base64
from supabase import create_client
import re
import numpy as np  # For image arrays
from scipy import ndimage  # For image processing
from scipy.ndimage import gaussian_filter
from enhance_nationality import enhance_nationality_with_ai
from extract_photos_v2 import (
    extract_embedded_image_candidates_from_pdf_page,
    extract_displayed_image_candidates_from_pdf_page,
    render_pdf_page_to_jpeg,
    crop_image_with_padding,
    normalize_face_crop_to_jpeg,
)

# cv2 is not needed - using PIL + scipy instead
CV2_AVAILABLE = True  # We can do face detection with PIL + scipy
CV2_FALLBACK_AVAILABLE = False

app = FastAPI(title="CV Parser Service", version="2.2.0-multi-format-support")

# In-memory LRU cache for document categorization (keyed on SHA-256 of file bytes)
# Prevents redundant OpenAI calls when the same passport/CNIC is re-submitted.
_DOC_CACHE_MAX_SIZE = 500
_doc_categorization_cache: OrderedDict = OrderedDict()

def _doc_cache_key(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _doc_cache_get(key: str):
    if key in _doc_categorization_cache:
        _doc_categorization_cache.move_to_end(key)
        return _doc_categorization_cache[key]
    return None

def _doc_cache_set(key: str, value: dict):
    if key in _doc_categorization_cache:
        _doc_categorization_cache.move_to_end(key)
    else:
        if len(_doc_categorization_cache) >= _DOC_CACHE_MAX_SIZE:
            _doc_categorization_cache.popitem(last=False)
    _doc_categorization_cache[key] = value

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

# Vision parsing page limits (kept conservative; can be tuned per env)
CV_VISION_MAX_PAGES = int(os.getenv("CV_VISION_MAX_PAGES", "2"))
CV_VISION_MAX_PAGES_FALLBACK = int(os.getenv("CV_VISION_MAX_PAGES_FALLBACK", "4"))
CV_VISION_MODEL = os.getenv("CV_VISION_MODEL", "gpt-4o-mini")
# Disable the second-pass vision retry (saves ~50% of CV parsing cost)
DISABLE_CV_VISION_RETRY = os.getenv("DISABLE_CV_VISION_RETRY", "false").lower() == "true"
# Disable ALL vision-based CV parsing (zero OpenAI vision cost; text-only parsing only)
DISABLE_CV_VISION = os.getenv("DISABLE_CV_VISION", "false").lower() == "true"
# Disable OpenAI Vision for document categorization (use text-only path instead)
# Set to true to stop OpenAI Vision calls on the /categorize-document endpoint
DISABLE_DOCUMENT_VISION = os.getenv("DISABLE_DOCUMENT_VISION", "false").lower() == "true"

# File extensions that are immediately rejected without any API calls
# These are Windows shortcuts, binaries, archives, and other non-document files
UNSUPPORTED_EXTENSIONS = {
    '.lnk', '.exe', '.dll', '.bat', '.cmd', '.msi', '.scr',
    '.zip', '.rar', '.7z', '.tar', '.gz',
    '.mp4', '.mp3', '.avi', '.mov', '.mkv', '.wav', '.flac',
    '.psd', '.ai', '.indd', '.sketch',
    '.db', '.sqlite', '.sql',
}
# Google Cloud Vision — service account JSON stored as env var string.
# Cost: $1.50/1000 pages vs OpenAI Vision $15/1000 pages (10x cheaper).
# Set GOOGLE_APPLICATION_CREDENTIALS_JSON to the full contents of your service account JSON file.
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
# Keep legacy API key support as fallback
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")
GOOGLE_VISION_ENABLED = bool(GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_VISION_API_KEY)
DOCUMENT_VISION_MAX_PAGES = int(os.getenv("DOCUMENT_VISION_MAX_PAGES", "2"))
DOCUMENT_TEXT_MIN_CHARS = int(os.getenv("DOCUMENT_TEXT_MIN_CHARS", "150"))
DOCUMENT_VISION_MODEL = os.getenv("DOCUMENT_VISION_MODEL", "gpt-4o-mini")
CV_PARSE_TEXT_INPUT_LIMIT = int(os.getenv("CV_PARSE_TEXT_INPUT_LIMIT", "6000"))

# Normalize SUPABASE_URL for storage3 client which expects a trailing slash
# Keep a clean version without trailing slash for public URL generation
if SUPABASE_URL:
    SUPABASE_URL_CLEAN = SUPABASE_URL.rstrip('/')
    SUPABASE_URL_FOR_CLIENT = SUPABASE_URL_CLEAN + '/'
else:
    SUPABASE_URL_CLEAN = SUPABASE_URL
    SUPABASE_URL_FOR_CLIENT = SUPABASE_URL

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
    use_textract: Optional[bool] = None  # True=use AWS Textract when configured; False=Vision-only; None=auto (use if AWS set)
    exhaustive_page_scan: Optional[bool] = None

class ParseResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class ExtractPhotoRequest(BaseModel):
    file_content: str
    attachment_id: str
    file_name: Optional[str] = None
    mime_type: Optional[str] = None


class ExtractPhotoResponse(BaseModel):
    success: bool
    profile_photo_url: Optional[str] = None
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


@app.on_event("startup")
def check_supabase_credentials():
    """Startup check: verify Supabase env and storage access."""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            logger.warning("[STARTUP] Supabase env vars missing: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
            return

        logger.info(f"[STARTUP] SUPABASE_URL={SUPABASE_URL}")

        # Quick storage access test
        try:
            client = create_client(SUPABASE_URL_FOR_CLIENT, SUPABASE_SERVICE_ROLE_KEY)
            # Attempt to list the root of the documents bucket
            try:
                _ = client.storage.from_('documents').list()
                logger.info('[STARTUP] Supabase storage access OK (documents bucket)')
            except Exception as se:
                logger.error(f"[STARTUP] Supabase storage access FAILED: {se}")
        except Exception as e:
            logger.error(f"[STARTUP] Failed to initialize Supabase client: {e}")
    except Exception as e:
        logger.error(f"[STARTUP] Unexpected error during Supabase credential check: {e}")

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
- cnic: Pakistani CNIC (National ID Card)
- driving_license: Driving license or driver's license
- police_character_certificate: Police character certificate or clearance certificate
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
- nationality: Nationality/citizenship (e.g., Pakistani, Indian, etc.) - CRITICAL RULES:
  * Extract ONLY the person's citizenship/nationality, NOT the country they worked in or want to work in
  * Look for "Nationality:", "Citizenship:", "Country of Origin:", or in passport/CNIC documents
  * If document is Pakistani CNIC (category = "cnic"), nationality MUST be "Pakistani" - do not extract from other fields
  * If document is Pakistani Passport (category = "passport" AND passport_no starts with "PA" or "AB"), nationality MUST be "Pakistani"
  * For CVs: Extract nationality ONLY from personal information section labeled "Nationality:" or "Citizenship:" - do NOT use work experience locations or country_of_interest
  * If you see "Worked in Saudi Arabia" or "Seeking opportunities in UAE", that is NOT nationality - those are work locations or country_of_interest
- passport_expiry: Passport expiry date (format: DD-MM-YYYY or YYYY-MM-DD)
- expiry_date: Alternative field for passport expiry date
- issue_date: Passport issue date (format: DD-MM-YYYY or YYYY-MM-DD)
- place_of_issue: Place where passport was issued (e.g., Islamabad, Karachi)
- country_of_interest: Country of interest/destination (ONLY for police_character_certificate documents - extract the country mentioned in the certificate, e.g., "UAE", "Saudi Arabia", "Qatar"). For other documents, leave null.

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
        
        # Post-processing: Move courses from experience to certifications BEFORE normalization
        experience_array = parsed_data.get('experience', [])
        existing_certifications = parsed_data.get('certifications', [])
        if not isinstance(existing_certifications, list):
            existing_certifications = []
        
        internship_keywords = [
            'intern', 'internship', 'internee', 'trainee', 'training', 'apprentice'
        ]
        course_keywords = [
            'course', 'certification', 'certificate', 'workshop', 'seminar', 
            'student', 'coursework', 'online course', 'certification program',
            'diploma', 'certification course', 'professional course', 'training program'
        ]
        
        filtered_experience = []
        internships_list = []
        
        if isinstance(experience_array, list):
            for exp in experience_array:
                if not isinstance(exp, dict):
                    filtered_experience.append(exp)
                    continue
                
                title = (exp.get('title') or '').lower()
                company = (exp.get('company') or '').lower()
                description = (exp.get('description') or '').lower()
                full_text = f"{title} {company} {description}".lower()

                # Check if this is an internship
                is_internship = any(k in title for k in internship_keywords) or \
                               any(k in company for k in internship_keywords) or \
                               any(k in description for k in internship_keywords)
                
                # Check if this is a course/training - check for multi-word keywords too
                is_course = False
                for keyword in course_keywords:
                    if keyword in full_text:
                        is_course = True
                        break

                # Handle internships and courses separately
                if is_internship:
                    internship_title = f"{exp.get('title', 'Internship')} at {exp.get('company', 'Unknown')}"
                    if internship_title not in internships_list:
                        internships_list.append(internship_title)
                elif is_course:
                    # Build cert title - prefer full entry or title+company combination
                    cert_title = None
                    if exp.get('title'):
                        cert_title = exp.get('title')
                        if exp.get('company') and exp.get('company') not in cert_title:
                            cert_title = f"{cert_title} ({exp.get('company')})"
                    else:
                        cert_title = exp.get('company')
                    
                    if cert_title and cert_title not in existing_certifications:
                        existing_certifications.append(cert_title)
                else:
                    # Keep real work experience
                    filtered_experience.append(exp)
        
        # Update parsed_data with cleaned experience and certifications
        parsed_data['experience'] = filtered_experience
        if existing_certifications:
            parsed_data['certifications'] = existing_certifications
        if internships_list:
            parsed_data['internships'] = internships_list
        
        # Now validate other fields
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
        
        # CRITICAL: Validate nationality based on document type
        # Pakistani CNIC or Pakistani Passport MUST have nationality = "Pakistani"
        if category in ['cnic', 'passport']:
            # Check if this is a Pakistani document
            is_pakistani_doc = False
            
            # Check CNIC format (Pakistani CNIC format: 12345-1234567-1 or 13 digits)
            cnic_value = identity_fields.get("cnic") or ""
            if cnic_value:
                # Remove dashes and spaces
                cnic_clean = cnic_value.replace("-", "").replace(" ", "")
                if len(cnic_clean) == 13 and cnic_clean.isdigit():
                    is_pakistani_doc = True
            
            # Check passport number (Pakistani passports usually start with PA, AB, or similar)
            passport_value = identity_fields.get("passport_no") or ""
            if passport_value:
                passport_upper = passport_value.upper().strip()
                if passport_upper.startswith("PA") or passport_upper.startswith("AB"):
                    is_pakistani_doc = True
            
            # Check place of issue (Pakistani cities)
            place_of_issue = identity_fields.get("place_of_issue") or ""
            pakistani_cities = ["islamabad", "karachi", "lahore", "peshawar", "quetta", "multan", "faisalabad", "rawalpindi"]
            if place_of_issue.lower() in pakistani_cities:
                is_pakistani_doc = True
            
            # If document type is CNIC or Passport and it appears to be Pakistani, enforce nationality
            if is_pakistani_doc or category == 'cnic':
                if identity_fields.get("nationality") and identity_fields.get("nationality").lower() != "pakistani":
                    logger.warning(f"Overriding nationality from '{identity_fields.get('nationality')}' to 'Pakistani' for {category} document")
                identity_fields["nationality"] = "Pakistani"
            
            # Special handling for Police Character Certificate: Extract country_of_interest
            if category == 'police_character_certificate':
                # Extract country_of_interest from document content if not already set
                if not identity_fields.get("country_of_interest"):
                    # Check common patterns in PCC documents
                    content_lower = content.lower()
                    pcc_countries = {
                        'uae': 'UAE',
                        'united arab emirates': 'UAE',
                        'dubai': 'UAE',
                        'saudi arabia': 'Saudi Arabia',
                        'ksa': 'Saudi Arabia',
                        'qatar': 'Qatar',
                        'kuwait': 'Kuwait',
                        'bahrain': 'Bahrain',
                        'oman': 'Oman'
                    }
                    for keyword, country in pcc_countries.items():
                        if keyword in content_lower:
                            identity_fields["country_of_interest"] = country
                            logger.info(f"Extracted country_of_interest '{country}' from Police Character Certificate")
                            break
        
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

def build_cv_prompt(content: str, from_images: bool = False) -> str:
    # Keep the parser payload intentionally bounded so input-token spend stays predictable.
    content_section = "CV Images are attached." if from_images else f"CV Content:\n{content[:CV_PARSE_TEXT_INPUT_LIMIT]}"
    return f"""
You are a CV/Resume parser. Extract structured information from the following CV content.
Return ONLY valid JSON with these exact fields (use null for missing data):

{{
  "full_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "nationality": "string or null (country of origin/citizenship) - CRITICAL: Extract ONLY from 'Nationality:' or 'Citizenship:' field in personal info section. Do NOT use work experience locations (e.g., 'Worked in Saudi Arabia' is NOT nationality). Do NOT use country_of_interest. If CNIC format is Pakistani (13 digits) or Passport starts with PA/AB, nationality MUST be 'Pakistani'.",
  "position": "string or null (desired job position/profession/title) - Normalize driver positions: HTV Driver, Heavy Duty Driver, Light Vehicle Driver, Simple Driver should all be normalized to 'Driver' or 'Driver (HTV)' if heavy vehicle",
  "experience_years": "number or null (total years of professional work experience)",
  "country_of_interest": "string or null (country they want to work in) - Extract from: 1) Objective/career goals (e.g., 'seeking opportunities in UAE'), 2) Work experience locations (if candidate worked in Gulf countries like Saudi Arabia, UAE, Qatar, Kuwait, Bahrain, Oman, that indicates country_of_interest). Do NOT confuse with nationality.",
  "linkedin_url": "string or null",
  "summary": "string or null",
  "professional_summary": "string or null (2-3 sentence career summary)",
  "father_name": "string or null (look for 'Father Name:', 'Father's Name:', 'Father:', or in personal information section)",
  "cnic": "string or null (Pakistani CNIC number, format: 12345-1234567-1 or 13 digits)",
  "passport": "string or null (passport number, look for 'Passport:', 'Passport No:', 'Passport Number:')",
  "date_of_birth": "string or null (date of birth, look for 'DOB:', 'Date of Birth:', 'Birth Date:', format: DD-MM-YYYY or YYYY-MM-DD)",
  "marital_status": "string or null (look for 'Marital Status:', 'Status:', 'Married', 'Single', etc.)",
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
      "degree": "string (REQUIRED - e.g., 'BS Electrical Engineering', 'Bachelor of Science', 'Intermediate', 'Matric', 'Diploma')",
      "institution": "string (REQUIRED - full university/college name, e.g., 'COMSATS UNIVERSITY ISLAMABAD')",
      "location": "string or null (city and country, e.g., 'Abbottabad, Pakistan')",
      "graduation_date": "string or null (year or date range, e.g., '2020-2024', '2024', 'Aug 2024')",
            "cgpa": "string or null (GPA/CGPA if mentioned, e.g., '3.01/4.00', '3.5')",
            "thesis": "string or null (if mentioned, keep verbatim)"
    }}
  ],
    "licenses": [
        {{
            "authority": "string (e.g., 'CORU', 'Allied Health Professionals Council Pakistan', 'Pakistan Physical Therapy Association (PPTA)')",
            "registration_no": "string or null (keep numbers verbatim)",
            "country": "string or null",
            "expiry_date": "string or null (year/date if available)",
            "notes": "string or null"
        }}
    ],
  "certifications": ["array of strings"],
  "languages": ["array of strings"],
  "previous_employment": "string or null (brief summary of work history)",
  "passport_expiry": "string or null (format: YYYY-MM-DD)",
  "gcc_years": "number or null (total years of work experience in GCC countries: Saudi Arabia, UAE, Qatar, Kuwait, Bahrain, Oman) - Calculate from experience array by summing years worked in GCC locations"
}}

IMPORTANT Guidelines:
- Extract nationality from personal info section, look for "Nationality:", "Country:", or in passport details
- Extract father_name from personal information section, look for "Father Name:", "Father's Name:", "Father:"
- Extract CNIC from personal information, look for "CNIC:", "CNIC #:", "ID Card:", format is usually 12345-1234567-1
- Extract passport number from personal information, look for "Passport:", "Passport No:", "Passport Number:"
- Extract date_of_birth from personal information, look for "DOB:", "Date of Birth:", "Birth Date:", "D.O.B:"
- Extract marital_status from personal information, look for "Marital Status:", "Status:", or keywords like "Married", "Single", "Divorced"
- Extract position from objective, desired role, or most recent job title
- Calculate experience_years from work history timeline if not stated
- Extract country_of_interest from: 1) Objective/goal statements (e.g., "seeking opportunities in UAE"), 2) Work experience locations (if worked in Gulf countries, that's country_of_interest, not nationality)
- Calculate gcc_years: Sum all years worked in GCC countries (Saudi Arabia, UAE, Qatar, Kuwait, Bahrain, Oman) from experience array
- Normalize driver positions: HTV Driver, Heavy Duty Driver, Light Vehicle Driver, Simple Driver → "Driver" or "Driver (HTV)"
- Extract ALL skills mentioned (technical, soft skills, software, languages, certifications)
- For skills, include programming languages, tools, frameworks, soft skills
- Be thorough in skills extraction - don't miss any mentioned abilities
- Pay special attention to the "PERSONAL INFORMATION" or "Personal Details" section for identity fields

CERTIFICATIONS EXTRACTION RULES (CRITICAL):
- EXTRACT certifications from MULTIPLE sources:
  1. Dedicated "CERTIFICATIONS" or "CERTIFICATES" sections
  2. "ONLINE COURSES", "ONLINE CERTIFICATIONS", "COURSERA", "UDEMY", "LINKEDIN LEARNING" sections
  3. Professional certifications like "PMP", "AWS", "Azure", "CCNA", "IELTS", etc.
  4. Any mention of courses, training programs, workshops, seminars (NOT work experience roles)
- Format: Include course/certification name AND provider/organization if available (e.g., "Power System Modelling and Fault Analysis (L&T EduTech)")
- Include dates/years when available
- Keep as complete strings in the certifications array (do NOT break into separate fields)
- CRITICAL: Do NOT include work experience roles in certifications - only actual learning/training activities
- If a section is labeled "ONLINE COURSES" or "COURSERA COURSES", extract ALL items from that section into certifications

EDUCATION EXTRACTION RULES (CRITICAL):
- ALWAYS extract education even if formatting is unusual
- Look for sections labeled: "EDUCATION", "ACADEMIC", "QUALIFICATION", "ACADEMIC BACKGROUND", "EDUCATIONAL BACKGROUND"
- Common degree formats: "BS", "B.Sc", "Bachelor of Science", "Masters", "M.Sc", "Intermediate", "Matric", "Diploma", "FSc", "FA"
- University names often in CAPS or title case (e.g., "COMSATS UNIVERSITY ISLAMABAD", "University of Engineering and Technology")
- Location usually follows university name (e.g., "Abbottabad, Pakistan", "Lahore, Pakistan")
- Dates can be: year ranges (2020-2024), single year (2024), or month-year (Aug 2024)
- CGPA/GPA often in format: "CGPA: 3.01/4.00", "GPA: 3.5/4.0"
- If education section exists but is unclear, extract whatever text is present rather than returning empty array
- NEVER return empty education array if any education text is visible in CV

INTERNSHIPS EXTRACTION RULES (IMPORTANT):
- EXTRACT internships from "INTERNSHIPS", "INTERNSHIP EXPERIENCE", "TRAINEE POSITIONS", "PROFESSIONAL TRAINING" sections
- Look for keywords: "Intern", "Internee", "Trainee", "Training", "Apprentice"
- Include internship title, company/organization, dates, and description
- CRITICAL: Do NOT include paid work experience (full-time/part-time jobs) in internships - ONLY unpaid/learning roles
- Distinction: If role has "Intern", "Internee", "Trainee", or "Training" in title/company → internship; otherwise → previous_employment
Format in experience array as regular experience entries, they will be filtered to internships array in post-processing

{content_section}

Return only the JSON object, no explanation.
"""

def post_process_cv_parsed_data(parsed_data: dict) -> dict:
    # Post-processing: Move courses from experience to certifications BEFORE normalization
    experience_array = parsed_data.get('experience', [])
    existing_certifications = parsed_data.get('certifications', [])
    if not isinstance(existing_certifications, list):
        existing_certifications = []
    
    internship_keywords = [
        'intern', 'internship', 'internee', 'trainee', 'training', 'apprentice'
    ]
    course_keywords = [
        'course', 'certification', 'certificate', 'workshop', 'seminar', 
        'coursework', 'online course', 'certification program',
        'diploma', 'certification course', 'professional course', 'training program'
    ]
    
    filtered_experience = []
    internships_list = []
    
    if isinstance(experience_array, list):
        for exp in experience_array:
            if not isinstance(exp, dict):
                filtered_experience.append(exp)
                continue
            
            title = (exp.get('title') or '').lower()
            company = (exp.get('company') or '').lower()
            description = (exp.get('description') or '').lower()
            full_text = f"{title} {company} {description}".lower()

            # Check if this is an internship
            is_internship = any(k in title for k in internship_keywords) or \
                           any(k in company for k in internship_keywords) or \
                           any(k in description for k in internship_keywords)
            
            # Check if this is a course/training - check for multi-word keywords too
            is_course = False
            for keyword in course_keywords:
                if keyword in full_text:
                    is_course = True
                    break

            # Handle internships and courses separately
            if is_internship:
                internship_title = f"{exp.get('title', 'Internship')} at {exp.get('company', 'Unknown')}"
                if internship_title not in internships_list:
                    internships_list.append(internship_title)
            elif is_course:
                # Build cert title - prefer full entry or title+company combination
                cert_title = None
                if exp.get('title'):
                    cert_title = exp.get('title')
                    if exp.get('company') and exp.get('company') not in cert_title:
                        cert_title = f"{cert_title} ({exp.get('company')})"
                else:
                    cert_title = exp.get('company')
                
                if cert_title and cert_title not in existing_certifications:
                    existing_certifications.append(cert_title)
            else:
                # Keep real work experience
                filtered_experience.append(exp)
    
    # Update parsed_data with cleaned experience and certifications
    parsed_data['experience'] = filtered_experience
    if existing_certifications:
        parsed_data['certifications'] = existing_certifications
    if internships_list:
        parsed_data['internships'] = internships_list

    return parsed_data

def looks_placeholder_cv(parsed_data: dict) -> bool:
    full_name = (parsed_data.get('full_name') or '').strip().lower()
    email = (parsed_data.get('email') or '').strip().lower()
    phone = (parsed_data.get('phone') or '').strip()

    if full_name in {'john doe', 'jane doe'}:
        return True
    if email.endswith('@example.com') or email.startswith('test@'):
        return True
    digits = ''.join(ch for ch in phone if ch.isdigit())
    if digits == '1234567890':
        return True
    return False

async def parse_cv_with_openai(content: str, filename: str) -> dict:
    """Parse CV content using OpenAI"""
    try:
        prompt = build_cv_prompt(content, from_images=False)

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise CV parser that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3500
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
        parsed_data = post_process_cv_parsed_data(parsed_data)

        # Evidence/audit trail for downstream completeness guards.
        # Keep truncated to avoid oversized payloads.
        if isinstance(content, str) and content:
            parsed_data['raw_text_length'] = len(content)
            parsed_data['raw_text'] = content[:CV_PARSE_TEXT_INPUT_LIMIT]
        
        # Post-processing: Normalize driver positions
        position = parsed_data.get('position', '').strip() if parsed_data.get('position') else ''
        if position:
            position_lower = position.lower()
            # Normalize all driver variants to "Driver"
            driver_variants = ['htv driver', 'heavy duty driver', 'heavy vehicle driver', 
                             'light vehicle driver', 'light duty driver', 'simple driver', 
                             'driver', 'truck driver', 'bus driver', 'van driver']
            if any(variant in position_lower for variant in driver_variants):
                # Keep sub-type if present (e.g., "HTV Driver" -> "Driver (HTV)")
                if 'htv' in position_lower or 'heavy' in position_lower:
                    parsed_data['position'] = 'Driver (HTV)'
                elif 'light' in position_lower:
                    parsed_data['position'] = 'Driver (Light Vehicle)'
                else:
                    parsed_data['position'] = 'Driver'
                logger.info(f"Normalized position from '{position}' to '{parsed_data['position']}'")

        # Keep education as a structured array (do not collapse to string).
        education_entries = parsed_data.get('education')
        if isinstance(education_entries, str) and education_entries.strip():
            parsed_data['education'] = [
                {
                    'degree': education_entries.strip(),
                    'institution': '',
                    'location': None,
                    'graduation_date': None,
                    'cgpa': None,
                    'thesis': None,
                }
            ]
        elif education_entries is None:
            parsed_data['education'] = []
        elif not isinstance(education_entries, list):
            parsed_data['education'] = []
        
        # Post-processing: Calculate GCC years from work experience
        gcc_countries = ['saudi arabia', 'uae', 'united arab emirates', 'qatar', 'kuwait', 
                        'bahrain', 'oman', 'gcc', 'gulf']
        experience_array = parsed_data.get('experience', [])
        gcc_years = 0
        
        if isinstance(experience_array, list):
            for exp in experience_array:
                if not isinstance(exp, dict):
                    continue
                location = (exp.get('location') or '').lower()
                start_date = exp.get('start_date')
                end_date = exp.get('end_date') or 'Present'
                
                # Check if location is in GCC
                is_gcc = any(gcc_country in location for gcc_country in gcc_countries)
                
                if is_gcc and start_date:
                    try:
                        # Try to parse dates and calculate years
                        # Simple year extraction (format: YYYY-MM-DD or YYYY)
                        start_year = None
                        end_year = None
                        
                        if '-' in start_date:
                            start_year = int(start_date.split('-')[0])
                        elif len(start_date) == 4 and start_date.isdigit():
                            start_year = int(start_date)
                        
                        if end_date.lower() == 'present' or end_date.lower() == 'current':
                            end_year = datetime.now().year
                        elif '-' in end_date:
                            end_year = int(end_date.split('-')[0])
                        elif len(end_date) == 4 and end_date.isdigit():
                            end_year = int(end_date)
                        
                        if start_year and end_year:
                            years = max(0, end_year - start_year)
                            gcc_years += years
                    except (ValueError, AttributeError):
                        # If date parsing fails, estimate 1 year per GCC job
                        gcc_years += 1
        
        if gcc_years > 0:
            parsed_data['gcc_years'] = gcc_years
            logger.info(f"Calculated GCC years: {gcc_years} from work experience")

        # Post-processing: Calculate total experience_years from work experience timeline
        # Note: experience_array already has courses and internships removed (done early)
        total_experience_years = 0
        experience_array = parsed_data.get('experience', [])
        
        if isinstance(experience_array, list):
            for exp in experience_array:
                if not isinstance(exp, dict):
                    continue

                start_date = exp.get('start_date')
                end_date = exp.get('end_date') or 'Present'

                if not start_date:
                    # No explicit dates available - do not guess duration.
                    continue

                try:
                    start_year = None
                    end_year = None

                    if '-' in start_date:
                        start_year = int(start_date.split('-')[0])
                    elif len(start_date) == 4 and start_date.isdigit():
                        start_year = int(start_date)

                    if isinstance(end_date, str) and (end_date.lower() == 'present' or end_date.lower() == 'current'):
                        end_year = datetime.now().year
                    elif isinstance(end_date, str) and '-' in end_date:
                        end_year = int(end_date.split('-')[0])
                    elif isinstance(end_date, str) and len(end_date) == 4 and end_date.isdigit():
                        end_year = int(end_date)

                    if start_year and end_year:
                        total_experience_years += max(0, end_year - start_year)
                except (ValueError, AttributeError):
                    # Date parsing failed - do not guess duration.
                    continue

        if total_experience_years > 0:
            try:
                existing_experience = float(parsed_data.get('experience_years')) if parsed_data.get('experience_years') is not None else None
            except (TypeError, ValueError):
                existing_experience = None

            if existing_experience is None:
                parsed_data['experience_years'] = total_experience_years
                logger.info(f"Calculated total experience_years: {total_experience_years} from work experience")
        
        # Post-processing: Enhance country_of_interest from work experience
        if not parsed_data.get('country_of_interest') or parsed_data.get('country_of_interest') == 'missing':
            # Extract from work experience locations
            countries_found = set()
            for exp in experience_array:
                if isinstance(exp, dict):
                    location = exp.get('location', '')
                    if location:
                        location_lower = location.lower()
                        # Map common variations
                        if 'saudi' in location_lower or 'ksa' in location_lower:
                            countries_found.add('Saudi Arabia')
                        elif 'uae' in location_lower or 'united arab emirates' in location_lower or 'dubai' in location_lower:
                            countries_found.add('UAE')
                        elif 'qatar' in location_lower:
                            countries_found.add('Qatar')
                        elif 'kuwait' in location_lower:
                            countries_found.add('Kuwait')
                        elif 'bahrain' in location_lower:
                            countries_found.add('Bahrain')
                        elif 'oman' in location_lower:
                            countries_found.add('Oman')
            
            if countries_found:
                # Use most common or first country
                parsed_data['country_of_interest'] = list(countries_found)[0]
                logger.info(f"Extracted country_of_interest from work experience: {parsed_data['country_of_interest']}")
        
        # Add "missing" default for country_of_interest if null or empty
        if not parsed_data.get('country_of_interest'):
            parsed_data['country_of_interest'] = 'missing'
        
        # ENHANCED: Use AI to infer nationality from education/work experience if not explicitly stated
        try:
            parsed_data = enhance_nationality_with_ai(parsed_data)
            if parsed_data.get('nationality_inferred_from'):
                logger.info(f"Nationality inference: {parsed_data.get('nationality')} (from {parsed_data.get('nationality_inferred_from')})")
        except Exception as e:
            logger.warning(f"Could not enhance nationality detection: {e}")
        
        logger.info(f"Successfully parsed CV: {filename}")
        return parsed_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, response: {result_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse OpenAI response as JSON: {str(e)}")
    except Exception as e:
        logger.error(f"OpenAI parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"CV parsing failed: {str(e)}")

async def parse_cv_with_vision(file_content: bytes, filename: str, max_pages: int = CV_VISION_MAX_PAGES) -> dict:
    """Parse CV using OpenAI Vision when text extraction is insufficient"""
    try:
        import base64

        def render_pdf_pages(pages_to_render: int):
            images_local = []
            pdf_doc_local = fitz.open(stream=file_content, filetype="pdf")
            total_pages_local = len(pdf_doc_local)
            pages_to_render_local = max(1, min(pages_to_render, total_pages_local))

            for page_index in range(pages_to_render_local):
                page = pdf_doc_local[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                jpeg_buf = io.BytesIO()
                img_pil.save(jpeg_buf, format="JPEG", quality=85, optimize=True)
                img_base64 = base64.b64encode(jpeg_buf.getvalue()).decode('utf-8')
                images_local.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
                logger.info(f"[CVVision] Rendered page {page_index + 1}/{pages_to_render_local} for vision parsing ({len(jpeg_buf.getvalue())//1024}KB JPEG)")

            pdf_doc_local.close()
            return images_local, total_pages_local, pages_to_render_local

        def strip_code_fences(text: str) -> str:
            t = text.strip()
            if t.startswith("```json"):
                t = t[7:]
            if t.startswith("```"):
                t = t[3:]
            if t.endswith("```"):
                t = t[:-3]
            return t.strip()

        def is_placeholder_text(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            lower = value.strip().lower()
            return lower in {"missing", "null", "undefined", "n/a", "na", "none", "not provided"}

        def needs_more_pages(parsed: dict) -> bool:
            exp = parsed.get('experience')
            exp_empty = not isinstance(exp, list) or len(exp) == 0
            prev = parsed.get('previous_employment')
            prev_missing = (not prev) or is_placeholder_text(prev)
            years = parsed.get('experience_years')
            years_missing = years is None or years == 0
            return exp_empty and (prev_missing or years_missing)

        async def run_vision_parse(images_payload):
            prompt = build_cv_prompt("", from_images=True)
            logger.info(f"[CVVision] Calling OpenAI Vision API ({CV_VISION_MODEL}) with {len(images_payload)} image(s)")
            response = openai.chat.completions.create(
                model=CV_VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise CV parser that returns only valid JSON."},
                    {"role": "user", "content": [{"type": "text", "text": prompt}, *images_payload]},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            result_text_local = strip_code_fences(response.choices[0].message.content or "")
            parsed_local = json.loads(result_text_local)
            return post_process_cv_parsed_data(parsed_local)

        images, total_pages, rendered_pages = render_pdf_pages(max_pages)
        if not images:
            raise HTTPException(status_code=400, detail="No pages found in PDF for vision parsing")

        parsed_data = await run_vision_parse(images)

        # If experience extraction is empty and the PDF has more pages, do a second pass with more pages.
        # Disabled via DISABLE_CV_VISION_RETRY=true to reduce OpenAI cost.
        if not DISABLE_CV_VISION_RETRY and needs_more_pages(parsed_data) and total_pages > rendered_pages and CV_VISION_MAX_PAGES_FALLBACK > rendered_pages:
            second_pages = min(CV_VISION_MAX_PAGES_FALLBACK, total_pages)
            logger.info(f"[CVVision] Experience appears missing; retrying with {second_pages} pages (pdf has {total_pages})")
            images2, _, rendered2 = render_pdf_pages(second_pages)
            if images2 and rendered2 > rendered_pages:
                parsed_data2 = await run_vision_parse(images2)
                # Prefer the second pass if it has more experience entries.
                exp1 = parsed_data.get('experience') if isinstance(parsed_data.get('experience'), list) else []
                exp2 = parsed_data2.get('experience') if isinstance(parsed_data2.get('experience'), list) else []
                if len(exp2) > len(exp1):
                    parsed_data = parsed_data2

        # ENHANCED: Use AI to infer nationality from education/work experience if not explicitly stated
        try:
            parsed_data = enhance_nationality_with_ai(parsed_data)
            if parsed_data.get('nationality_inferred_from'):
                logger.info(f"Vision parsing - Nationality inference: {parsed_data.get('nationality')} (from {parsed_data.get('nationality_inferred_from')})")
        except Exception as e:
            logger.warning(f"Could not enhance nationality detection in vision parsing: {e}")

        return parsed_data

    except Exception as e:
        logger.error(f"OpenAI vision parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"CV vision parsing failed: {str(e)}")


async def parse_cv_with_vision_image(file_content: bytes, filename: str, image_format: str) -> dict:
    """
    Parse CV directly from image file (JPEG, PNG, GIF, WebP, BMP) using OpenAI Vision
    No PDF conversion needed - sends image directly to Vision API
    
    Args:
        file_content: Raw image bytes
        filename: Original filename (for logging)
        image_format: Image format ('jpeg', 'png', 'gif', 'webp', 'bmp')
    
    Returns:
        Parsed CV data dictionary
    """
    try:
        import base64
        
        logger.info(f"[CVVisionImage] Processing {image_format.upper()} image: {filename}")
        
        # Helper functions
        def strip_code_fences(text: str) -> str:
            t = text.strip()
            if t.startswith("```json"):
                t = t[7:]
            if t.startswith("```"):
                t = t[3:]
            if t.endswith("```"):
                t = t[:-3]
            return t.strip()
        
        # Encode image as base64
        img_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Map format to MIME type
        mime_type_map = {
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }
        mime_type = mime_type_map.get(image_format, 'image/jpeg')
        
        # Create image payload for Vision API
        images = [{
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
        }]
        
        logger.info(f"[CVVisionImage] Encoded {image_format.upper()} image ({len(img_base64)} chars base64)")
        
        # Build CV parsing prompt
        prompt = build_cv_prompt("", from_images=True)
        
        # Call OpenAI Vision API
        logger.info(f"[CVVisionImage] Calling OpenAI Vision API ({CV_VISION_MODEL})")
        response = openai.chat.completions.create(
            model=CV_VISION_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise CV parser that returns only valid JSON."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, *images]},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        
        result_text = strip_code_fences(response.choices[0].message.content or "")
        parsed_data = json.loads(result_text)
        parsed_data = post_process_cv_parsed_data(parsed_data)
        
        # ENHANCED: Use AI to infer nationality from education/work experience if not explicitly stated
        try:
            parsed_data = enhance_nationality_with_ai(parsed_data)
            if parsed_data.get('nationality_inferred_from'):
                logger.info(f"[CVVisionImage] Nationality inference: {parsed_data.get('nationality')} (from {parsed_data.get('nationality_inferred_from')})")
        except Exception as e:
            logger.warning(f"[CVVisionImage] Could not enhance nationality detection: {e}")
        
        logger.info(f"[CVVisionImage] Successfully parsed CV from {image_format.upper()} image")
        return parsed_data
        
    except Exception as e:
        logger.error(f"[CVVisionImage] Image CV parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"CV image parsing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CV Parser",
        "version": "2.2.0-multi-format-support",
        "status": "running",
        "environment": ENVIRONMENT,
        "features": [
            "embedded-image-extraction",
            "jpeg-conversion",
            "enhanced-logging",
            "multi-format-cv-support",
            "pdf-image-fallback",
            "direct-image-processing"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.2.0-multi-format-support",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_configured": bool(OPENAI_API_KEY),
        "hmac_configured": bool(HMAC_SECRET),
        "photo_jpeg_enabled": True,
        "embedded_image_extraction": True,
        "supported_formats": ["pdf", "jpeg", "png", "gif", "webp", "bmp"]
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
        file_bytes = base64.b64decode(categorize_request.file_content)

        # Check in-memory cache first (avoids re-calling OpenAI for the same document bytes)
        _ck = _doc_cache_key(file_bytes)
        _cached = _doc_cache_get(_ck)
        if _cached:
            logger.info(f"[DocumentCategorization] Cache hit for {categorize_request.file_name} (sha256={_ck[:16]}...) - skipping OpenAI call")
            return _cached

        # Use OpenAI Vision API to read documents directly (much more reliable than text extraction)
        # Vision API accepts: PNG, JPEG, GIF, WebP (NOT PDFs directly)
        # So we need to:
        # 1. For PDFs: Convert pages to images, then send images
        # 2. For images: Send directly (just encode as base64)
        # 3. For text files: Extract text and send to text-based API
        
        mime_type = categorize_request.mime_type or "application/pdf"
        
        if mime_type == "application/pdf":
            # PDF: Convert pages to images, then send to Vision API
            logger.info(f"[DocumentCategorization] PDF detected - converting to images for Vision API: {categorize_request.file_name}")
            result = await categorize_document_with_vision_api(file_bytes, categorize_request.file_name, is_pdf=True)
        elif mime_type in ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]:
            # Image: Send directly to Vision API (just encode as base64)
            logger.info(f"[DocumentCategorization] Image detected - sending directly to Vision API: {categorize_request.file_name}")
            result = await categorize_document_with_vision_api(file_bytes, categorize_request.file_name, is_pdf=False)
        else:
            # Text files: Extract text and use text-based API
            try:
                file_text = file_bytes.decode('utf-8', errors='ignore')
            except:
                file_text = str(file_bytes)
            
            logger.info(f"[DocumentCategorization] Text file - extracted {len(file_text)} characters from {categorize_request.file_name}")
            result = await categorize_document_with_ai_text(file_text, categorize_request.file_name, mime_type)
        
        # Map extracted_identity to identity_fields for backward compatibility
        # Vision API returns extracted_identity, but backend expects identity_fields
        if result.get('extracted_identity'):
            identity_fields = result['extracted_identity']
            # Ensure all required fields are present
            identity_fields = {
                "name": identity_fields.get("name"),
                "father_name": identity_fields.get("father_name"),
                "cnic": identity_fields.get("cnic"),
                "passport_no": identity_fields.get("passport_no"),
                "email": identity_fields.get("email"),
                "phone": identity_fields.get("phone"),
                "date_of_birth": identity_fields.get("date_of_birth") or identity_fields.get("dob"),
                "dob": identity_fields.get("dob") or identity_fields.get("date_of_birth"),
                "document_number": identity_fields.get("document_number"),
                "nationality": identity_fields.get("nationality"),
                "passport_expiry": identity_fields.get("passport_expiry") or identity_fields.get("expiry_date"),
                "expiry_date": identity_fields.get("expiry_date") or identity_fields.get("passport_expiry"),
                "issue_date": identity_fields.get("issue_date"),
                "place_of_issue": identity_fields.get("place_of_issue")
            }
            result['identity_fields'] = identity_fields
        else:
            # No identity fields extracted - return empty structure
            result['identity_fields'] = {
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
        
        # Always return required fields, even if missing
        _response = {
            "success": True,
            "category": result.get("category", "other_documents"),
            "confidence": float(result.get("confidence", 0.0)),
            "identity_fields": result.get("identity_fields")
        }
        _doc_cache_set(_ck, _response)
        return _response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/split-and-categorize")
async def split_and_categorize(
    request: Request,
    categorize_request: CategorizeRequest,
    x_hmac_signature: str = Header(None),
):
    """
    Dual OCR + Vision fallback. PDF -> pages -> Engine A (Textract+Vision) or B (Vision-only).
    Returns { success, engine_used, documents }.
    engine_used: "vision_only" | "textract+vision". Per doc: doc_type, pages, split_strategy, confidence, identity, pdf_base64, needs_review.
    """
    if not x_hmac_signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")
    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    try:
        file_bytes = base64.b64decode(categorize_request.file_content)
        mime_type = categorize_request.mime_type or "application/pdf"
        is_pdf = mime_type == "application/pdf"
        if not is_pdf and mime_type not in ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]:
            raise HTTPException(status_code=400, detail="Unsupported type. Use PDF or image (jpeg, png, gif, webp).")

        result = await run_split_and_categorize(
            file_content=file_bytes,
            file_name=categorize_request.file_name,
            is_pdf=is_pdf,
            openai_api_key=OPENAI_API_KEY,
            candidate_data=categorize_request.candidate_data,
            use_textract=categorize_request.use_textract,
            exhaustive_page_scan=bool(categorize_request.exhaustive_page_scan),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SplitAndCategorize] Error: {e}")
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

def detect_file_type(file_content: bytes) -> str:
    """
    Detect file type from magic bytes (file header signature)
    Returns: 'pdf', 'jpeg', 'png', 'gif', 'webp', 'bmp', or 'unknown'
    """
    if len(file_content) < 4:
        return 'unknown'
    
    # Check magic bytes
    header = file_content[:4]
    
    # PDF: %PDF (0x25 0x50 0x44 0x46)
    if header[0] == 0x25 and header[1] == 0x50 and header[2] == 0x44 and header[3] == 0x46:
        return 'pdf'
    
    # JPEG: FF D8
    if header[0] == 0xFF and header[1] == 0xD8:
        return 'jpeg'
    
    # PNG: 89 50 4E 47
    if header[0] == 0x89 and header[1] == 0x50 and header[2] == 0x4E and header[3] == 0x47:
        return 'png'
    
    # GIF: 47 49 46
    if header[0] == 0x47 and header[1] == 0x49 and header[2] == 0x46:
        return 'gif'
    
    # WebP: check for RIFF + WEBP signature
    if len(file_content) >= 12:
        if (header[0] == 0x52 and header[1] == 0x49 and header[2] == 0x46 and header[3] == 0x46 and
            file_content[8] == 0x57 and file_content[9] == 0x45 and file_content[10] == 0x42 and file_content[11] == 0x50):
            return 'webp'
    
    # BMP: 42 4D
    if header[0] == 0x42 and header[1] == 0x4D:
        return 'bmp'
    
    return 'unknown'


@app.post("/parse-cv")
async def parse_cv_from_url(
    request: Request,
    x_signature: str = Header(None)
):
    """
    Parse CV from URL - backend worker endpoint with smart fallback
    
    Supports:
    - PDF files (text extraction → Vision fallback)
    - Image files (JPEG, PNG, GIF, WebP, BMP) → Direct Vision API processing
    """

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

        # Reject unsupported file types immediately (no API calls)
        _cv_file_name = payload.get('file_name', file_url.split('/')[-1].split('?')[0])
        _cv_ext = os.path.splitext(_cv_file_name.lower())[1] if _cv_file_name else ''
        if _cv_ext in UNSUPPORTED_EXTENSIONS:
            logger.info(f"[CVParse] Rejected unsupported file type '{_cv_ext}': {_cv_file_name}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {_cv_ext}. Only PDF and image files are accepted.")

        # Fetch file from URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_content = response.content

        # Detect file type
        file_type = detect_file_type(file_content)
        logger.info(f"[CVParse] Detected file type: {file_type}")

        # Initialize variables
        text_content = ""
        profile_photo_url = None
        used_vision = False
        parsed_data = None

        # Strategy 1: Try PDF extraction if it's a PDF
        if file_type == 'pdf':
            try:
                text_content = extract_text_from_pdf(file_content)
                
                # PHASE C COMPLETE: Real ML-based face detection enabled!
                profile_photo_url = await extract_profile_photo_from_pdf(file_content, attachment_id or "unknown")
                
                # Parse with OpenAI (fallback to Vision when text extraction is weak)
                if len(text_content.strip()) < 200:
                    logger.warning(f"[CVParse] Low text extracted ({len(text_content)} chars). Trying Google Vision OCR...")
                    google_text = await extract_text_from_scanned_pdf_google_vision(file_content)
                    if google_text and len(google_text.strip()) > 100:
                        logger.info(f"[CVParse] Google Vision OCR extracted {len(google_text)} chars. Parsing with text model.")
                        parsed_data = await parse_cv_with_openai(google_text, attachment_id or "unknown")
                    else:
                        logger.warning("[CVParse] Google Vision OCR insufficient. Parsing with available text (no OpenAI Vision).")
                        parsed_data = await parse_cv_with_openai(text_content or google_text or "", attachment_id or "unknown")
                else:
                    parsed_data = await parse_cv_with_openai(text_content, attachment_id or "unknown")
                    
            except Exception as pdf_error:
                logger.error(f"[CVParse] PDF extraction failed: {pdf_error}")
                # Try Google Vision OCR as recovery path
                google_text = await extract_text_from_scanned_pdf_google_vision(file_content)
                if google_text and len(google_text.strip()) > 50:
                    logger.info(f"[CVParse] Google Vision OCR recovered {len(google_text)} chars after PDF failure.")
                    try:
                        parsed_data = await parse_cv_with_openai(google_text, attachment_id or "unknown")
                    except Exception as ocr_parse_error:
                        logger.error(f"[CVParse] OpenAI text parse after Google OCR failed: {ocr_parse_error}")
                        raise HTTPException(status_code=500, detail=f"Failed to parse OCR text: {str(ocr_parse_error)}")
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse PDF with text extraction and Google Vision OCR: {str(pdf_error)}"
                    )
        
        # Strategy 2: If it's an image, use Google Vision OCR and then parse the extracted text.
        elif file_type in ['jpeg', 'png', 'gif', 'webp', 'bmp']:
            logger.info(f"[CVParse] Image file detected ({file_type}).")
            profile_photo_url = await extract_profile_photo_from_image(file_content, attachment_id or "unknown")
            
            if GOOGLE_VISION_ENABLED:
                # Use Google Vision OCR (cheap, ~$1.50/1000 pages)
                google_text = await _google_vision_ocr_image_bytes(file_content)
                if google_text and len(google_text.strip()) > 50:
                    logger.info(f"[CVParse] Google Vision OCR extracted {len(google_text)} chars from image CV.")
                    parsed_data = await parse_cv_with_openai(google_text, attachment_id or "unknown")
                else:
                    logger.warning("[CVParse] Google Vision OCR returned insufficient text for image CV.")
                    raise HTTPException(
                        status_code=422,
                        detail="Could not extract text from image CV using Google Vision OCR. Please re-upload as PDF."
                    )
            else:
                raise HTTPException(
                    status_code=422,
                    detail="Image-format CVs require Google Cloud Vision (GOOGLE_APPLICATION_CREDENTIALS_JSON not configured). Please re-upload as PDF."
                )
        
        # Strategy 3: Unknown file type - try Google Vision OCR before failing.
        else:
            logger.warning(f"[CVParse] Unknown file type. Attempting Google Vision OCR as last resort...")
            google_text = await extract_text_from_scanned_pdf_google_vision(file_content)
            if not google_text or len(google_text.strip()) < 50:
                google_text = await _google_vision_ocr_image_bytes(file_content)
            if google_text and len(google_text.strip()) > 50:
                logger.info(f"[CVParse] Google Vision OCR recovered {len(google_text)} chars from unknown file type.")
                parsed_data = await parse_cv_with_openai(google_text, attachment_id or "unknown")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Please upload a PDF or image file (JPEG, PNG, GIF, WebP, BMP)."
                )
        
        # Ensure we have parsed data
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Failed to extract any CV data")
        
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

async def categorize_document_with_vision_api(file_content: bytes, file_name: str, is_pdf: bool = True) -> dict:
    """
    Use OpenAI Vision API to read documents directly - much more reliable than text extraction!
    
    IMPORTANT: OpenAI Vision API only accepts image formats (PNG, JPEG, GIF, WebP) - NOT PDFs!
    That's why we need to:
    - For PDFs: Convert pages to images first (using PyMuPDF)
    - For images: Just encode as base64 and send directly
    
    Args:
        file_content: PDF bytes or image bytes
        file_name: Original filename
        is_pdf: True if PDF, False if already an image
    """
    try:
        import base64
        images = []
        
        if is_pdf:
            # PDF: Convert pages to images (REQUIRED - Vision API doesn't accept PDFs)
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            total_pages = len(pdf_doc)
            pages_to_process = min(total_pages, DOCUMENT_VISION_MAX_PAGES)
            
            for page_num in range(pages_to_process):
                page = pdf_doc[page_num]
                # Render page to image (1.5x zoom = ~108 DPI — sufficient for GPT-4o-mini vision)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                
                # Convert to JPEG for smaller payload (saves egress vs PNG)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85, optimize=True)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
                
                logger.info(f"[VisionAPI] Converted PDF page {page_num + 1}/{pages_to_process} to image ({len(img_base64)} chars base64)")
            
            pdf_doc.close()
            
            if not images:
                raise Exception("No pages found in PDF")
            
            logger.info(
                f"[VisionAPI] Converted {len(images)} of {total_pages} PDF page(s) to images (cap={DOCUMENT_VISION_MAX_PAGES})"
            )
        else:
            # Image: Send directly (just encode as base64 - no conversion needed!)
            img_base64 = base64.b64encode(file_content).decode('utf-8')
            
            # Determine image format from filename or content
            image_format = "png"
            if file_name.lower().endswith(('.jpg', '.jpeg')):
                image_format = "jpeg"
            elif file_name.lower().endswith('.gif'):
                image_format = "gif"
            elif file_name.lower().endswith('.webp'):
                image_format = "webp"
            
            images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{img_base64}"
                }
            })
            
            logger.info(f"[VisionAPI] Encoded image directly as base64 ({len(img_base64)} chars, format: {image_format})")
        
        logger.info(f"[VisionAPI] Sending {len(images)} image(s) to OpenAI Vision API")
        
        # Prepare prompt for Vision API
        prompt = """You are a document classification and identity extraction AI. Analyze the document image(s) and provide:

1. Document category (choose ONE):
   - cv_resume: CV, resume, curriculum vitae
   - passport: Passport copy, passport scan
   - certificates: Educational certificates, degrees, diplomas, training certificates
   - contracts: Employment contracts, offer letters, agreements
   - medical_reports: Medical test reports, health certificates, fitness certificates
   - photos: Passport photos, ID photos
   - other_documents: Any other document type

2. Confidence score (0.0 to 1.0) for the category classification

3. Extract ALL identity fields from the document. READ THE DOCUMENT CAREFULLY and extract:
   - name: Full name of the person (look for "Full Name:", "Name:", "Full Name", etc.)
   - father_name: Father's name (common in Pakistani documents)
   - cnic: Pakistani CNIC number (format: 12345-1234567-1 or 13 digits)
   - passport_no: Passport number (look for "Passport No:", "Passport Number:", "Passport No", etc. - e.g., PA1234567, AB1234567)
   - email: Email address
   - phone: Phone number
   - date_of_birth: Date of birth (look for "Date of Birth:", "DOB:", "Date of Birth", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - document_number: Any other ID number found in the document
   - nationality: Nationality (look for "Nationality:", "Country:", "Nationality", etc. - e.g., Pakistani, Indian, etc.)
   - passport_expiry: Passport expiry date (look for "Expiry Date:", "Expiry:", "Expiry Date", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - expiry_date: Alternative field for passport expiry date
   - issue_date: Passport issue date (look for "Issue Date:", "Issue Date", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - place_of_issue: Place where passport was issued (look for "Place of Issue:", "Place of Issue", etc. - e.g., Islamabad, Karachi)

Return ONLY valid JSON with this exact structure:
{
  "category": "category_name",
  "confidence": 0.95,
  "ocr_confidence": 0.90,
  "extracted_identity": {
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
  }
}

IMPORTANT: Read the document image(s) carefully and extract ALL visible information. The document may have clear text like:
- Full Name: Muhammad Farhan
- Passport No: PA1234567
- Nationality: Pakistani
- Date of Birth: 15-08-1994
- Issue Date: 10-06-2022
- Expiry Date: 09-06-2032
- Place of Issue: Islamabad

Extract this information even if labels are slightly different."""
        
        # Call OpenAI Vision API
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build messages with images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *images  # Add all page images
                ]
            }
        ]
        
        logger.info(f"[VisionAPI] Calling OpenAI Vision API ({DOCUMENT_VISION_MODEL}) with {len(images)} image(s)")
        
        response = client.chat.completions.create(
            model=DOCUMENT_VISION_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"[VisionAPI] Received response from OpenAI ({len(result_text)} chars)")
        
        # Clean up markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        parsed_result = json.loads(result_text)
        
        # Ensure extracted_identity exists
        extracted_identity = parsed_result.get('extracted_identity', {})
        if not extracted_identity:
            extracted_identity = {}
        
        # Ensure backward compatibility
        if extracted_identity.get("dob") and not extracted_identity.get("date_of_birth"):
            extracted_identity["date_of_birth"] = extracted_identity["dob"]
        
        if extracted_identity.get("expiry_date") and not extracted_identity.get("passport_expiry"):
            extracted_identity["passport_expiry"] = extracted_identity["expiry_date"]
        
        parsed_result['extracted_identity'] = extracted_identity
        
        non_null_identity = {k: v for k, v in extracted_identity.items() if v is not None} if extracted_identity else {}
        
        logger.info(f"[VisionAPI] Categorized as: {parsed_result.get('category')} (confidence: {parsed_result.get('confidence')})")
        if non_null_identity:
            logger.info(f"[VisionAPI] Extracted identity fields: {list(non_null_identity.keys())}")
            logger.info(f"[VisionAPI] Extracted values: {non_null_identity}")
        else:
            logger.warning(f"[VisionAPI] No identity fields extracted from document")
        
        return parsed_result
        
    except Exception as e:
        logger.error(f"[VisionAPI] Error: {e}")
        import traceback
        logger.error(f"[VisionAPI] Traceback: {traceback.format_exc()}")
        logger.warning(f"[VisionAPI] Falling back to OCR/text extraction method")

        if is_pdf:
            ocr_text = await extract_text_from_scanned_pdf_google_vision(file_content)
            if ocr_text and len(ocr_text.strip()) >= 50:
                logger.info(f"[VisionAPI] Google Vision OCR recovered {len(ocr_text)} chars from PDF {file_name}")
                return await categorize_document_with_ai_text(ocr_text, file_name, "application/pdf")

            logger.warning(f"[VisionAPI] Google Vision OCR unavailable or insufficient for PDF {file_name}; trying native PDF text extraction")
            text_content = extract_text_from_pdf(file_content)
            return await categorize_document_with_ai_text(text_content, file_name, "application/pdf")

        ocr_text = await _google_vision_ocr_image_bytes(file_content)
        if ocr_text and len(ocr_text.strip()) >= 50:
            logger.info(f"[VisionAPI] Google Vision OCR recovered {len(ocr_text)} chars from image {file_name}")
            return await categorize_document_with_ai_text(ocr_text, file_name, "image/jpeg")

        logger.warning(f"[VisionAPI] Google Vision OCR unavailable or insufficient for image {file_name}; returning unknown")
        return {
            "success": True,
            "category": "unknown",
            "confidence": 0.0,
            "ocr_confidence": 0.0,
            "extracted_identity": None,
            "raw_text": None,
        }

async def _google_vision_annotate_image_bytes(image_bytes: bytes, features: list[dict]) -> dict:
    """Call Google Cloud Vision annotate on a single in-memory image."""
    if not GOOGLE_VISION_ENABLED:
        return {}
    try:
        payload = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
                "features": features,
            }]
        }

        if GOOGLE_APPLICATION_CREDENTIALS_JSON:
            import google.oauth2.service_account as sa
            import google.auth.transport.requests as ga_requests
            creds = sa.Credentials.from_service_account_info(
                json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON),
                scopes=["https://www.googleapis.com/auth/cloud-vision"]
            )
            creds.refresh(ga_requests.Request())
            url = "https://vision.googleapis.com/v1/images:annotate"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {creds.token}"
            }
        else:
            url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
            headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=json.dumps(payload), headers=headers)

        if resp.status_code != 200:
            logger.warning(f"[GoogleVision] HTTP {resp.status_code}: {resp.text[:200]}")
            return {}
        return resp.json().get("responses", [{}])[0]
    except Exception as e:
        logger.warning(f"[GoogleVision] annotate call failed: {e}")
        return {}

async def _google_vision_ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Call Google Cloud Vision DOCUMENT_TEXT_DETECTION on a single in-memory image.
    Authenticates using service account JSON (GOOGLE_APPLICATION_CREDENTIALS_JSON env var)
    or falls back to API key (GOOGLE_VISION_API_KEY env var).
    """
    data = await _google_vision_annotate_image_bytes(
        image_bytes,
        [{"type": "DOCUMENT_TEXT_DETECTION"}],
    )
    return data.get("fullTextAnnotation", {}).get("text", "") if data else ""

async def _google_vision_face_detect_image_bytes(image_bytes: bytes) -> list[dict]:
    """Detect faces on a single in-memory image using Google Vision FACE_DETECTION."""
    data = await _google_vision_annotate_image_bytes(
        image_bytes,
        [{"type": "FACE_DETECTION", "maxResults": 5}],
    )
    return data.get("faceAnnotations", []) if data else []

async def extract_text_from_scanned_pdf_google_vision(pdf_content: bytes, max_pages: int = 3) -> str:
    """
    OCR a scanned/image-only PDF using Google Cloud Vision DOCUMENT_TEXT_DETECTION.
    Cost: ~$1.50/1000 pages vs OpenAI Vision ~$15/1000 pages (10x cheaper).
    Requires GOOGLE_APPLICATION_CREDENTIALS_JSON (service account) or GOOGLE_VISION_API_KEY env var.
    Returns extracted text, or empty string if credentials not configured or call fails.
    """
    if not GOOGLE_VISION_ENABLED:
        logger.info("[GoogleVision] No credentials configured, skipping Google Vision OCR")
        return ""

    try:
        import fitz  # PyMuPDF — already in requirements

        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(pdf_doc)
        pages_to_scan = min(max_pages, total_pages)
        logger.info(f"[GoogleVision] Scanning {pages_to_scan}/{total_pages} pages via Google Cloud Vision OCR")

        all_text_parts = []

        for page_num in range(pages_to_scan):
            page = pdf_doc[page_num]
            # Render at 150 DPI — good enough for OCR, keeps payload small
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            page_text = await _google_vision_ocr_image_bytes(img_bytes)
            logger.info(f"[GoogleVision] Page {page_num+1}: extracted {len(page_text)} chars")
            all_text_parts.append(page_text)

        pdf_doc.close()
        combined = "\n".join(all_text_parts)
        logger.info(f"[GoogleVision] Total extracted: {len(combined)} chars across {pages_to_scan} page(s)")
        return combined

    except Exception as e:
        logger.warning(f"[GoogleVision] OCR failed: {e}")
        return ""

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

def is_image_blurry(image: Image.Image, threshold: float = 100.0) -> bool:
    """
    Check if image is blurry using Laplacian variance method (via scipy).
    Lower variance = blurrier. Typical threshold: 100.
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array
        
        # Apply Laplacian filter
        laplacian = ndimage.laplace(gray.astype(float))
        laplacian_var = np.var(laplacian)
        
        is_blurry = laplacian_var < threshold
        logger.info(f"[QUALITY_CHECK] blur_check variance={laplacian_var:.1f} threshold={threshold} is_blurry={is_blurry}")
        return is_blurry
    except Exception as e:
        logger.warning(f"[QUALITY_CHECK] blur_check failed: {e}")
        return False

def is_image_too_dark_or_bright(image: Image.Image, dark_threshold: int = 30, bright_threshold: int = 220) -> tuple:
    """
    Check if image is too dark or too bright.
    Returns (is_bad, reason).
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)
        
        mean_brightness = np.mean(gray)
        
        if mean_brightness < dark_threshold:
            logger.info(f"[QUALITY_CHECK] brightness_check brightness={mean_brightness:.1f} reason=TOO_DARK")
            return True, "TOO_DARK"
        if mean_brightness > bright_threshold:
            logger.info(f"[QUALITY_CHECK] brightness_check brightness={mean_brightness:.1f} reason=TOO_BRIGHT")
            return True, "TOO_BRIGHT"
        
        logger.info(f"[QUALITY_CHECK] brightness_check brightness={mean_brightness:.1f} status=GOOD")
        return False, "OK"
    except Exception as e:
        logger.warning(f"[QUALITY_CHECK] brightness_check failed: {e}")
        return False, "UNKNOWN"


def _extract_face_bounds(face: dict, image_width: int, image_height: int) -> Optional[tuple[int, int, int, int, float]]:
    poly = face.get("fdBoundingPoly") or face.get("boundingPoly") or {}
    vertices = poly.get("vertices") or []
    if not vertices:
        return None

    xs = [max(0, min(image_width, int(vertex.get("x", 0) or 0))) for vertex in vertices]
    ys = [max(0, min(image_height, int(vertex.get("y", 0) or 0))) for vertex in vertices]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    if right <= left or bottom <= top:
        return None

    area_ratio = ((right - left) * (bottom - top)) / max(1, image_width * image_height)
    return left, top, right, bottom, area_ratio


def _normalize_angle(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _face_angles_are_acceptable(face: dict) -> bool:
    pan = abs(_normalize_angle(face.get("panAngle")))
    roll = abs(_normalize_angle(face.get("rollAngle")))
    tilt = abs(_normalize_angle(face.get("tiltAngle")))
    return pan <= 40 and roll <= 35 and tilt <= 25


def _portrait_score(width: int, height: int) -> float:
    aspect = width / max(1, height)
    return max(0.0, 1.0 - min(1.0, abs(aspect - 1.0) / 0.75))


def _evaluate_face_crop(image_bytes: bytes, face: dict, source: str, label: str) -> Optional[dict]:
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        image_width, image_height = pil_img.size

        bounds = _extract_face_bounds(face, image_width, image_height)
        if not bounds:
            return None

        left, top, right, bottom, area_ratio = bounds
        detection_confidence = float(face.get("detectionConfidence") or 0.0)
        if detection_confidence < 0.65:
            return None
        if not _face_angles_are_acceptable(face):
            return None

        if source == 'embedded':
            if area_ratio < 0.08 or area_ratio > 0.85:
                return None
        else:
            if area_ratio < 0.01 or area_ratio > 0.20:
                return None

        face_crop = crop_image_with_padding(image_bytes, left, top, right, bottom, padding_ratio=0.30)
        if face_crop.width < 96 or face_crop.height < 96:
            return None
        if is_image_blurry(face_crop, threshold=100):
            return None

        is_bad_brightness, brightness_reason = is_image_too_dark_or_bright(face_crop)
        if is_bad_brightness:
            logger.info(f"[PHOTO_EXTRACT] label={label} action=REJECT reason={brightness_reason}")
            return None

        size_score = min(1.0, max(face_crop.width, face_crop.height) / 512.0)
        portrait_score = _portrait_score(face_crop.width, face_crop.height)
        score = (detection_confidence * 0.65) + (size_score * 0.20) + (portrait_score * 0.15)

        return {
            "score": score,
            "crop": face_crop,
            "confidence": detection_confidence,
            "area_ratio": area_ratio,
            "label": label,
        }
    except Exception as exc:
        logger.warning(f"[PHOTO_EXTRACT] label={label} action=REJECT reason=EVAL_ERROR error={exc}")
        return None


async def _select_best_face_crop(image_bytes: bytes, source: str, label: str) -> Optional[dict]:
    faces = await _google_vision_face_detect_image_bytes(image_bytes)
    if not faces:
        logger.info(f"[PHOTO_EXTRACT] label={label} action=SKIP reason=NO_FACES_DETECTED")
        return None

    best = None
    for face in faces:
        candidate = _evaluate_face_crop(image_bytes, face, source, label)
        if candidate and (best is None or candidate["score"] > best["score"]):
            best = candidate
    return best


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pil_image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffer, format='JPEG', quality=95, optimize=True)
    return buffer.getvalue()


async def _select_avatar_from_wide_banner(image_bytes: bytes, label: str) -> Optional[dict]:
    """
    Some CV templates flatten the header into one wide banner image that contains
    the circular avatar plus surrounding text/icons. Try to recover the avatar
    region from that banner before giving up on direct image extraction.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        image_w, image_h = pil_img.size
        face_like_regions = detect_faces_with_mediapipe(pil_img)
        if face_like_regions:
            face_like_regions = sorted(face_like_regions, key=lambda item: item[4], reverse=True)
            x, y, w, h, confidence = face_like_regions[0]
        else:
            photo_region = detect_photo_region_heuristic(pil_img)
            if not photo_region:
                return None
            x, y, w, h, confidence = photo_region

        crop = crop_image_with_padding(image_bytes, int(x), int(y), int(x + w), int(y + h), padding_ratio=0.22)
        if crop.width < 96 or crop.height < 96:
            return None
        if is_image_blurry(crop, threshold=45):
            return None

        is_bad_brightness, _ = is_image_too_dark_or_bright(crop, dark_threshold=20, bright_threshold=245)
        if is_bad_brightness:
            return None

        crop_bytes = _pil_image_to_jpeg_bytes(crop)
        face_candidate = await _select_best_face_crop(crop_bytes, 'embedded', f"{label} derived_crop")
        face_bonus = face_candidate['score'] if face_candidate else 0.0

        aspect = crop.width / max(1, crop.height)
        square_bias = 1.0 - _clamp01(abs(aspect - 1.0) / 0.40)
        size_bias = min(1.0, max(crop.width, crop.height) / 320.0)
        score = (confidence * 0.40) + (square_bias * 0.30) + (size_bias * 0.15) + (face_bonus * 0.15)

        return {
            'score': score,
            'crop': crop,
            'confidence': face_bonus,
            'label': f"{label} derived_crop",
            'square_bias': square_bias,
            'top_bias': 1.0,
        }
    except Exception as exc:
        logger.warning(f"[PHOTO_EXTRACT] label={label} action=REJECT reason=WIDE_BANNER_EVAL_ERROR error={exc}")
        return None


async def _select_best_displayed_avatar_candidate(candidates: list, label_prefix: str) -> Optional[dict]:
    """
    Score displayed image blocks using page placement and image geometry first,
    then add any face-detection confidence as a bonus instead of requiring it.
    """
    best = None

    for candidate in candidates:
        if candidate.width < 80 or candidate.height < 80:
            continue
        if candidate.width > 3000 or candidate.height > 3000:
            continue
        if not candidate.bbox or not candidate.page_width or not candidate.page_height:
            continue

        x0, y0, x1, y1 = candidate.bbox
        bbox_w = max(1.0, float(x1) - float(x0))
        bbox_h = max(1.0, float(y1) - float(y0))
        page_area_ratio = (bbox_w * bbox_h) / max(1.0, candidate.page_width * candidate.page_height)
        if page_area_ratio < 0.005 or page_area_ratio > 0.18:
            continue

        aspect = candidate.width / max(1, candidate.height)
        if aspect < 0.65:
            continue

        is_wide_banner = aspect > 1.35 and (float(y0) / candidate.page_height) < 0.20 and page_area_ratio >= 0.08
        if aspect > 1.35 and not is_wide_banner:
            continue

        if is_wide_banner:
            banner_candidate = await _select_avatar_from_wide_banner(
                candidate.image_bytes,
                f"{label_prefix} block={candidate.image_index}"
            )
            if banner_candidate and (best is None or banner_candidate['score'] > best['score']):
                best = banner_candidate
            continue

        top_bias = 1.0 - _clamp01((float(y0) / candidate.page_height) / 0.40)
        square_bias = 1.0 - _clamp01(abs(aspect - 1.0) / 0.35)
        size_bias = 1.0 - _clamp01(abs(page_area_ratio - 0.045) / 0.06)

        pil_img = Image.open(io.BytesIO(candidate.image_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        if is_image_blurry(pil_img, threshold=55):
            logger.info(f"[PHOTO_EXTRACT] label={label_prefix} block={candidate.image_index} action=REJECT reason=BLURRY_DISPLAYED_IMAGE")
            continue

        is_bad_brightness, brightness_reason = is_image_too_dark_or_bright(pil_img, dark_threshold=20, bright_threshold=245)
        if is_bad_brightness:
            logger.info(f"[PHOTO_EXTRACT] label={label_prefix} block={candidate.image_index} action=REJECT reason={brightness_reason}")
            continue

        face_candidate = await _select_best_face_crop(
            candidate.image_bytes,
            'embedded',
            f"{label_prefix} block={candidate.image_index}"
        )
        face_bonus = face_candidate['score'] if face_candidate else 0.0

        score = (top_bias * 0.38) + (square_bias * 0.27) + (size_bias * 0.20) + (face_bonus * 0.15)

        evaluated = {
            'score': score,
            'crop': pil_img,
            'confidence': face_bonus,
            'label': f"{label_prefix} block={candidate.image_index}",
            'page_area_ratio': page_area_ratio,
            'top_bias': top_bias,
            'square_bias': square_bias,
        }
        if best is None or evaluated['score'] > best['score']:
            best = evaluated

    return best

def detect_photo_region_heuristic(image: Image.Image) -> Optional[tuple]:
    """
    Heuristic: Find likely photo region on document page using simple brightness analysis.
    
    Returns:
        (x, y, w, h, confidence) of most likely photo region, or None
    """
    try:
        w, h = image.size
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)
        
        # Search in top 40% of page (where photos usually are)
        search_height = int(h * 0.4)
        search_region = gray[:search_height, :]
        
        # Find regions with good contrast (photos tend to have varied brightness)
        # Using simple edge detection via gradient
        if search_height > 10 and w > 10:
            # Calculate horizontal and vertical gradients
            gy = np.abs(np.diff(search_region, axis=0))
            gx = np.abs(np.diff(search_region, axis=1))
            
            # Edge magnitude
            edges = (np.pad(gy, ((0, 1), (0, 0)), mode='constant') + 
                    np.pad(gx, ((0, 0), (0, 1)), mode='constant'))
            
            # Find regions with high edge density
            window_size = 100
            if edges.shape[0] > window_size and edges.shape[1] > window_size:
                edge_density = ndimage.uniform_filter(edges, size=window_size)
                
                # Find peak edge density (likely photo region)
                max_idx = np.unravel_index(np.argmax(edge_density), edge_density.shape)
                if max_idx[0] > 0 and max_idx[1] > 0:
                    y = max(0, max_idx[0] - window_size // 2)
                    x = max(0, max_idx[1] - window_size // 2)
                    
                    # Confidence based on edge density
                    confidence = min(1.0, edge_density[max_idx] / 10.0)
                    if confidence > 0.3:
                        logger.info(f"[PHOTO_REGION] Detected at ({x},{y}) size={window_size}x{window_size} confidence={confidence:.2f}")
                        return (x, y, window_size, window_size, confidence)
        
        logger.info("[PHOTO_REGION] No obvious photo region detected")
        return None
        
    except Exception as e:
        logger.warning(f"[PHOTO_REGION] detection failed: {e}")
        return None

def detect_faces_with_mediapipe(image: Image.Image) -> list:
    """
    Detect potential face regions using simple brightness/contrast analysis.
    Since we don't have opencv, use heuristics: face regions typically have
    good contrast and specific brightness ranges.
    
    Returns:
        List of (x, y, w, h, confidence) for potential face regions
    """
    try:
        w, h = image.size
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)
        
        # Normalize to 0-1 range
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-5)
        
        # Calculate local contrast using gaussian blur difference
        smooth = gaussian_filter(gray, sigma=5)
        contrast = np.abs(gray - smooth)
        
        # Find regions with good contrast (faces have distinctive features)
        threshold = np.percentile(contrast, 75)
        high_contrast = contrast > threshold
        
        # Find connected regions of high contrast
        labeled, num_features = ndimage.label(high_contrast)
        
        faces = []
        min_size = 30
        max_size = min(w, h) // 2
        
        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            
            # Get bounding box
            rows = np.any(region, axis=1)
            cols = np.any(region, axis=0)
            if not rows.any() or not cols.any():
                continue
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            rw = xmax - xmin
            rh = ymax - ymin
            
            # Face-like region criteria
            if min_size < rw < max_size and min_size < rh < max_size:
                aspect_ratio = rw / rh if rh > 0 else 0
                # Faces are roughly square to slightly portrait (0.7-1.2)
                if 0.7 <= aspect_ratio <= 1.2:
                    # Confidence based on contrast and size
                    region_contrast = np.mean(contrast[region])
                    confidence = min(1.0, region_contrast * 2)
                    
                    if confidence > 0.4:
                        faces.append((xmin, ymin, rw, rh, confidence))
                        logger.info(f"[FACE_DETECT] Region at ({xmin},{ymin}) {rw}x{rh} confidence={confidence:.2f}")
        
        # Sort by confidence and keep top 3
        faces.sort(key=lambda f: f[4], reverse=True)
        faces = faces[:3]
        
        logger.info(f"[FACE_DETECT] Found {len(faces)} potential face regions")
        return faces
            
    except Exception as e:
        logger.warning(f"[FACE_DETECT] Face detection failed: {e}")
        return []

async def extract_profile_photo_from_image(image_content: bytes, attachment_id: str) -> Optional[str]:
    """
    Extract profile photo from image file (JPEG, PNG, etc.) using Google Vision face detection.

    Returns the signed URL of the uploaded photo, or None if no photo found.
    """
    try:
        logger.info(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} action=START")
        if not GOOGLE_VISION_ENABLED:
            logger.info(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} action=SKIP reason=GOOGLE_VISION_DISABLED")
            return None

        best = await _select_best_face_crop(image_content, 'image', f"candidate_id={attachment_id} source=image")
        if not best:
            return None

        photo_bytes = normalize_face_crop_to_jpeg(best['crop'])
        logger.info(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} face_ready score={best['score']:.2f}")
        
        # Upload to Supabase (will return signed URL)
        photo_url = upload_photo_to_supabase(photo_bytes, attachment_id, "jpg")
        
        if photo_url:
            logger.info(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} SUCCESS uploaded_as={photo_url}")
        else:
            logger.warning(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} action=SKIP reason=UPLOAD_FAILED")
        
        return photo_url
        
    except Exception as e:
        logger.warning(f"[PHOTO_EXTRACT_IMG] candidate_id={attachment_id} extraction_failed error={e}")
        import traceback
        logger.warning(f"[PHOTO_EXTRACT_IMG] traceback={traceback.format_exc()}")
        return None  # Graceful fallback


async def extract_profile_photo_from_pdf(pdf_content: bytes, attachment_id: str, max_pages: int = 5) -> Optional[str]:
    """
    Extract profile photo from PDF using Google Vision face detection.

    Returns the signed URL of the uploaded photo, or None if no photo found.
    """
    try:
        logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} action=START")
        if not GOOGLE_VISION_ENABLED:
            logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} action=SKIP reason=GOOGLE_VISION_DISABLED")
            return None

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        if pdf_document.page_count == 0:
            logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} action=SKIP reason=NO_PAGES")
            return None

        pages_to_scan = min(max_pages, pdf_document.page_count)
        pdf_document.close()

        best = None
        for page_index in range(pages_to_scan):
            displayed_candidates = extract_displayed_image_candidates_from_pdf_page(pdf_content, page_index)
            displayed_label = f"candidate_id={attachment_id} source=displayed_image_block page={page_index + 1}"
            displayed_best = await _select_best_displayed_avatar_candidate(displayed_candidates, displayed_label)
            if displayed_best and (best is None or displayed_best['score'] > best['score']):
                best = displayed_best
                if displayed_best['score'] >= 0.82:
                    break

            embedded_candidates = extract_embedded_image_candidates_from_pdf_page(pdf_content, page_index)
            for embedded in embedded_candidates:
                if embedded.width < 80 or embedded.height < 80:
                    continue
                if embedded.width > 3000 or embedded.height > 3000:
                    continue

                label = f"candidate_id={attachment_id} source=embedded page={page_index + 1} image={embedded.image_index}"
                candidate = await _select_best_face_crop(embedded.image_bytes, 'embedded', label)
                if candidate and (best is None or candidate['score'] > best['score']):
                    best = candidate
                    if candidate['score'] >= 0.88:
                        break
            if best and best['score'] >= 0.88:
                break

        if not best:
            for page_index in range(pages_to_scan):
                rendered_jpeg = render_pdf_page_to_jpeg(pdf_content, page_index, scale=2.0)
                label = f"candidate_id={attachment_id} source=rendered page={page_index + 1}"
                candidate = await _select_best_face_crop(rendered_jpeg, 'rendered_page', label)
                if candidate and (best is None or candidate['score'] > best['score']):
                    best = candidate
                    if candidate['score'] >= 0.88:
                        break

        if best is None or best['score'] < 0.72:
            logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} action=SKIP reason=NO_PHOTO_FOUND")
            return None

        photo_bytes = normalize_face_crop_to_jpeg(best['crop'])
        logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} face_ready score={best['score']:.2f}")
        
        # Upload to Supabase (will return signed URL)
        photo_url = upload_photo_to_supabase(photo_bytes, attachment_id, "jpg")
        
        if photo_url:
            logger.info(f"[PHOTO_EXTRACT] candidate_id={attachment_id} SUCCESS uploaded_as={photo_url}")
        else:
            logger.warning(f"[PHOTO_EXTRACT] candidate_id={attachment_id} action=SKIP reason=UPLOAD_FAILED")
        
        return photo_url
        
    except Exception as e:
        logger.warning(f"[PHOTO_EXTRACT] candidate_id={attachment_id} extraction_failed error={e}")
        import traceback
        logger.warning(f"[PHOTO_EXTRACT] traceback={traceback.format_exc()}")
        return None  # Graceful fallback - don't fail CV parsing if photo extraction fails


@app.post("/extract-photo", response_model=ExtractPhotoResponse)
async def extract_photo(
    request: Request,
    extract_request: ExtractPhotoRequest,
    x_hmac_signature: str = Header(None)
):
    """Extract only the profile photo from a PDF (base64 bytes) and return profile_photo_url."""

    if not x_hmac_signature:
        return ExtractPhotoResponse(success=False, error="Missing HMAC signature")

    body = await request.body()
    if not verify_hmac(x_hmac_signature, body):
        return ExtractPhotoResponse(success=False, error="Invalid HMAC signature")

    try:
        pdf_bytes = base64.b64decode(extract_request.file_content)
        url = await extract_profile_photo_from_pdf(pdf_bytes, extract_request.attachment_id)
        if not url:
            return ExtractPhotoResponse(success=True, profile_photo_url=None)
        return ExtractPhotoResponse(success=True, profile_photo_url=url)
    except Exception as e:
        logger.error(f"[ExtractPhoto] Error: {e}")
        return ExtractPhotoResponse(success=False, error=str(e))

def upload_photo_to_supabase(image_bytes: bytes, attachment_id: str, file_ext: str) -> str:
    """Upload extracted photo to Supabase Storage bucket using supabase-py client"""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            logger.warning("[PhotoUpload] Supabase credentials not configured, skipping photo upload")
            return None
        
        logger.info(f"[PhotoUpload] Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL_FOR_CLIENT, SUPABASE_SERVICE_ROLE_KEY)
        
        # Bucket and file path
        bucket_name = "documents"
        file_path = f"candidate_photos/{attachment_id}/profile.{file_ext}"
        
        logger.info(f"[PhotoUpload] Uploading to: {bucket_name}/{file_path}")
        logger.info(f"[PhotoUpload] Image size: {len(image_bytes):,} bytes")
        
        # Upload using Supabase client.
        # In production this may run multiple times for the same attachment_id;
        # avoid failing with a 409 Duplicate.
        content_type = "image/jpeg" if file_ext.lower() in ("jpg", "jpeg") else f"image/{file_ext.lower()}"
        file_options = {
            "content-type": content_type,
            # storage3 supports upsert; when enabled we overwrite existing objects.
            # If the backend ignores it, we still handle 409 Duplicate below.
            "upsert": "true",
        }

        response = None
        try:
            response = supabase.storage.from_(bucket_name).upload(file_path, image_bytes, file_options=file_options)
        except Exception as upload_err:
            # If object already exists, treat as success and return the existing public URL.
            # storage3 raises StorageApiError with payload: {'statusCode': 409, 'error': 'Duplicate', ...}
            err_text = str(upload_err)
            if "statusCode" in err_text and "409" in err_text and "Duplicate" in err_text:
                public_url = f"{SUPABASE_URL_CLEAN}/storage/v1/object/public/{bucket_name}/{file_path}"
                logger.warning(f"[PhotoUpload] Duplicate object; reusing existing URL: {public_url}")
                return public_url
            raise
        
        logger.info(f"[PhotoUpload] Upload response: {response}")
        
        # Return public URL (use clean URL without trailing slash)
        public_url = f"{SUPABASE_URL_CLEAN}/storage/v1/object/public/{bucket_name}/{file_path}"
        logger.info(f"[PhotoUpload] Success! Public URL: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"[PhotoUpload] Photo upload error: {e}")
        import traceback
        logger.error(f"[PhotoUpload] Traceback: {traceback.format_exc()}")
        return None

async def categorize_document_with_ai_text(text_content: str, file_name: str, mime_type: str) -> dict:
    """
    Use OpenAI to categorize document and extract identity fields from already-extracted text.
    
    This function receives text_content that has already been extracted from a PDF/image.
    It uses GPT to categorize the document and extract identity information.
    
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
        # Ensure text_content is a string
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content else ""
        
        logger.info(f"[DocumentCategorization] Processing text content - {len(text_content)} characters from {file_name}")
        
        # Truncate text_content if too large to avoid token limit errors
        # gpt-4o-mini has TPM limit of 200,000 tokens
        # Rough estimate: 1 token ≈ 4 characters, so ~50,000 chars ≈ 12,500 tokens
        # Leave room for prompt (~500 tokens) and response (~1000 tokens)
        MAX_TEXT_LENGTH = 50000  # ~12,500 tokens
        if len(text_content) > MAX_TEXT_LENGTH:
            logger.warning(f"[DocumentCategorization] Text content too large ({len(text_content)} chars), truncating to {MAX_TEXT_LENGTH} chars")
            text_content = text_content[:MAX_TEXT_LENGTH] + "\n\n[... content truncated due to size ...]"
        
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

3. Extract identity fields from the document. READ THE DOCUMENT CONTENT CAREFULLY and extract:
   - name: Full name of the person (look for "Full Name:", "Name:", "Full Name", etc.)
   - father_name: Father's name (common in Pakistani documents)
   - cnic: Pakistani CNIC number (format: 12345-1234567-1 or 13 digits)
   - passport_no: Passport number (look for "Passport No:", "Passport Number:", "Passport No", etc. - e.g., PA1234567, AB1234567)
   - email: Email address
   - phone: Phone number
   - date_of_birth: Date of birth (look for "Date of Birth:", "DOB:", "Date of Birth", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - document_number: Any other ID number found in the document
   - nationality: Nationality (look for "Nationality:", "Country:", "Nationality", etc. - e.g., Pakistani, Indian, etc.)
   - passport_expiry: Passport expiry date (look for "Expiry Date:", "Expiry:", "Expiry Date", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - expiry_date: Alternative field for passport expiry date
   - issue_date: Passport issue date (look for "Issue Date:", "Issue Date", etc. - format: DD-MM-YYYY or YYYY-MM-DD)
   - place_of_issue: Place where passport was issued (look for "Place of Issue:", "Place of Issue", etc. - e.g., Islamabad, Karachi)
   
   IMPORTANT: The document content is provided below. Extract the information even if labels are slightly different (e.g., "Full Name" vs "Name", "Passport No" vs "Passport Number").

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
                {"role": "system", "content": "You are a document classification and identity extraction expert. Carefully read the document content and extract ALL identity fields. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000  # Increased to ensure all fields are extracted
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

        # Reject unsupported file types immediately (no API calls)
        _ext = os.path.splitext(file_name.lower())[1] if file_name else ''
        if _ext in UNSUPPORTED_EXTENSIONS:
            logger.info(f"[CategorizeDocument] Rejected unsupported file type '{_ext}': {file_name}")
            return {
                "success": True,
                "category": "unsupported",
                "confidence": 1.0,
                "ocr_confidence": None,
                "extracted_identity": None,
                "mismatch_fields": [],
                "raw_text": None,
            }

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
        
        # Decode base64 to bytes
        try:
            file_bytes = base64.b64decode(file_content, validate=False)
        except Exception as decode_error:
            logger.error(f"[CategorizeDocument] Base64 decode error: {decode_error}")
            raise HTTPException(status_code=400, detail=f"Invalid base64-encoded file content: {str(decode_error)}")

        # Check in-memory cache first (avoids re-calling OpenAI for the same document bytes)
        _ck2 = _doc_cache_key(file_bytes)
        _cached2 = _doc_cache_get(_ck2)
        if _cached2:
            logger.info(f"[CategorizeDocument] Cache hit for {file_name} (sha256={_ck2[:16]}...) - skipping OpenAI call")
            return _cached2

        # Use OpenAI Vision API to read documents directly (much more reliable than text extraction)
        # Vision API accepts: PNG, JPEG, GIF, WebP (NOT PDFs directly)
        # So we need to:
        # 1. For PDFs: Convert pages to images, then send images
        # 2. For images: Send directly (just encode as base64)
        # 3. For text files: Extract text and send to text-based API
        
        if mime_type == "application/pdf":
            file_text = ""
            try:
                file_text = extract_text_from_pdf(file_bytes)
            except Exception as text_error:
                logger.warning(f"[CategorizeDocument] PDF text extraction failed for {file_name}: {text_error}")

            if len(file_text.strip()) >= DOCUMENT_TEXT_MIN_CHARS:
                logger.info(
                    f"[CategorizeDocument] PDF text extracted ({len(file_text)} chars). Using text categorization for {file_name}"
                )
                result = await categorize_document_with_ai_text(file_text, file_name, mime_type)
            elif DISABLE_DOCUMENT_VISION:
                # Vision disabled — return unknown rather than call OpenAI Vision
                logger.info(
                    f"[CategorizeDocument] PDF text weak ({len(file_text)} chars) and DISABLE_DOCUMENT_VISION=true. Returning unknown for {file_name}"
                )
                result = {"category": "unknown", "confidence": 0.0, "extracted_identity": None}
            else:
                logger.info(
                    f"[CategorizeDocument] PDF text weak ({len(file_text)} chars). Falling back to capped vision for {file_name}"
                )
                result = await categorize_document_with_vision_api(file_bytes, file_name, is_pdf=True)
        elif mime_type in ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]:
            if DISABLE_DOCUMENT_VISION:
                # Vision disabled — use Google Vision OCR text extraction then text API
                logger.info(f"[CategorizeDocument] Image detected, DISABLE_DOCUMENT_VISION=true. Using OCR+text path for {file_name}")
                ocr_text = ""
                if GOOGLE_VISION_ENABLED:
                    try:
                        import base64 as _b64
                        _img_b64 = _b64.b64encode(file_bytes).decode('utf-8')
                        ocr_text = await google_vision_ocr_image(_img_b64)
                    except Exception as ocr_err:
                        logger.warning(f"[CategorizeDocument] Google Vision OCR failed for {file_name}: {ocr_err}")
                if len(ocr_text.strip()) >= 50:
                    result = await categorize_document_with_ai_text(ocr_text, file_name, mime_type)
                else:
                    logger.info(f"[CategorizeDocument] Insufficient OCR text for {file_name}. Returning unknown.")
                    result = {"category": "unknown", "confidence": 0.0, "extracted_identity": None}
            else:
                # Image: Send directly to Vision API (just encode as base64)
                logger.info(f"[CategorizeDocument] Image detected - sending directly to Vision API: {file_name}")
                result = await categorize_document_with_vision_api(file_bytes, file_name, is_pdf=False)
        else:
            # Text files: Extract text and use text-based API
            try:
                file_text = file_bytes.decode('utf-8', errors='ignore')
            except:
                file_text = str(file_bytes)
            
            logger.info(f"[CategorizeDocument] Text file - extracted {len(file_text)} characters from {file_name}")
            result = await categorize_document_with_ai_text(file_text, file_name, mime_type)
        # Ensure all required fields are present, even if null
        _response2 = {
            "success": True,
            "category": result.get("category", None),
            "confidence": result.get("confidence", None),
            "ocr_confidence": result.get("ocr_confidence", None),
            "extracted_identity": result.get("extracted_identity", None),
            "mismatch_fields": result.get("mismatch_fields", []),
            "raw_text": None  # Don't return raw text to reduce payload size
        }
        _doc_cache_set(_ck2, _response2)
        return _response2
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
