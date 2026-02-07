"""
Unified document split & classification.

Dual OCR + Vision fallback:
- Primary: AWS Textract + OpenAI Vision (layout + classify).
- Fallback: OpenAI Vision only (layout inference + classify).
Engine chosen automatically; processing never fails due to Textract.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from textract_layout import (
        aws_configured,
        detect_engine,
        get_blocks_from_image,
        cluster_blocks_to_regions,
        validate_no_overlap,
        crop_image_to_region,
    )
except ImportError:
    aws_configured = lambda: False
    detect_engine = None
    get_blocks_from_image = None
    cluster_blocks_to_regions = None
    validate_no_overlap = None
    crop_image_to_region = None

CONFIDENCE_THRESHOLD = 0.88
DOC_CATEGORIES = [
    "cv_resume", 
    "passport", 
    "cnic", 
    "driving_license", 
    "police_character_certificate",
    "educational_documents",
    "experience_certificates",
    "navttc_reports",
    "certificates",  # Professional/IT certifications ONLY
    "contracts", 
    "medical_reports", 
    "photos", 
    "other_documents",
]

def normalize_doc_type(category: str) -> str:
    """
    Normalize category to canonical doc_type.
    Maps variations to canonical types for new categories.
    """
    if not category:
        return "other_documents"
    
    category_lower = category.lower().strip()
    
    # Map variations to canonical types
    normalization_map = {
        # CNIC / National ID
        "national_id": "cnic",
        "id_card": "cnic",
        "nid": "cnic",
        
        # Driving License
        "driving_license": "driving_license",
        "drivers_license": "driving_license",
        "driver_license": "driving_license",
        "drivers_licence": "driving_license",
        "driving_licence": "driving_license",
        "dl": "driving_license",
        
        # Police Certificate
        "character_certificate": "police_character_certificate",
        "police_clearance": "police_character_certificate",
        "pcc": "police_character_certificate",
        "character_clearance": "police_character_certificate",
        "police_certificate": "police_character_certificate",
        
        # Educational Documents
        "degree": "educational_documents",
        "diploma": "educational_documents",
        "transcript": "educational_documents",
        "marksheet": "educational_documents",
        "academic_certificate": "educational_documents",
        "educational_certificate": "educational_documents",
        "university_degree": "educational_documents",
        "college_diploma": "educational_documents",
        
        # Experience Certificates
        "experience_certificate": "experience_certificates",
        "employment_certificate": "experience_certificates",
        "experience_letter": "experience_certificates",
        "service_certificate": "experience_certificates",
        "employment_letter": "experience_certificates",
        "work_reference": "experience_certificates",
        
        # NAVTTC Reports
        "navttc": "navttc_reports",
        "navtic": "navttc_reports",
        "nvtc": "navttc_reports",
        "navttc_certificate": "navttc_reports",
        "vocational_certificate": "navttc_reports",
        "trade_certificate": "navttc_reports",
        "technical_training": "navttc_reports",
        
        # Professional Certificates
        "professional_certificate": "certificates",
        "skill_certificate": "certificates",
        "it_certificate": "certificates",
    }
    
    # Check if category needs normalization
    if category_lower in normalization_map:
        return normalization_map[category_lower]
    
    # Return as-is if it's already a canonical type
    if category_lower in DOC_CATEGORIES:
        return category_lower
    
    # Default fallback
    return "other_documents"

VISION_PROMPT = """You are a document classification and identity extraction AI. Analyze this SINGLE page image and provide:

1. Document category (choose ONE - BE SPECIFIC):
   
   🎓 EDUCATIONAL DOCUMENTS (academic qualifications):
   - educational_documents: University degrees (BSc, MSc, BA, MA, PhD), college diplomas, 
     academic transcripts, marksheets, school certificates, graduation certificates
   
   👷 NAVTTC VOCATIONAL REPORTS (government technical training):
   - navttc_reports: NAVTTC certificates, NAVTIC training reports, NVTC vocational certificates,
     government technical training, trade test certificates, skill development from NAVTTC
   
   💼 EXPERIENCE CERTIFICATES (employment proof):
   - experience_certificates: Employment certificates, experience letters, service certificates,
     work reference letters, relieving letters, employment verification, NOC from employer
   
   👮 POLICE CLEARANCE:
   - police_character_certificate: Police clearance certificate, character certificate,
     background check, PCC, police verification
   
   📜 PROFESSIONAL CERTIFICATES (skill/industry certifications):
   - certificates: CCNA, AWS, PMP, Microsoft certifications, Cisco certifications,
     professional licenses, industry skill certifications, IT certifications
     (NOT academic degrees, NOT NAVTTC, NOT employment letters)
   
   📄 OTHER CATEGORIES:
   - cv_resume: CV, resume, curriculum vitae
   - passport: Passport copy, passport scan
   - cnic: Pakistani CNIC (National ID Card)
   - driving_license: Driving license or driver's license
   - contracts: Employment contracts, offer letters, agreements
   - medical_reports: Medical test reports, health certificates, fitness certificates
   - photos: Passport photos, ID photos
   - other_documents: Any other document type

2. Confidence score (0.0 to 1.0) for the category classification.

3. Extract ALL identity fields from the document:
   - name, father_name, cnic, passport_no, email, phone
   - date_of_birth, document_number, nationality
   - passport_expiry, expiry_date, issue_date, place_of_issue
   
   CRITICAL NATIONALITY RULES:
   - If document category is "cnic" (Pakistani CNIC), nationality MUST be "Pakistani"
   - If document category is "passport" AND passport_no starts with "PA" or "AB", nationality MUST be "Pakistani"
   - Extract nationality ONLY from explicit "Nationality:" or "Citizenship:" fields
   - Do NOT extract nationality from work experience locations or country_of_interest
   - For CVs: If you see "Worked in Saudi Arabia", that is NOT nationality - that's work location

Return ONLY valid JSON:
{
  "category": "category_name",
  "confidence": 0.95,
  "extracted_identity": {
    "name": "string or null",
    "father_name": "string or null",
    "cnic": "string or null",
    "passport_no": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "date_of_birth": "string or null",
    "document_number": "string or null",
    "nationality": "string or null",
    "passport_expiry": "string or null",
    "expiry_date": "string or null",
    "issue_date": "string or null",
    "place_of_issue": "string or null"
  }
}
"""

VISION_LAYOUT_PROMPT = """You are a document layout and classification AI. Analyze this page image.

1. LAYOUT: Is this page a SINGLE document (whole page) or MULTIPLE distinct documents (e.g. passport top, medical bottom)?
   - "single": one document fills the page
   - "multi": 2+ clearly separate documents in bands/sections (top-half, bottom-half, etc.)

2. If "single": give category, confidence, extracted_identity (same as standard classification).

3. If "multi": list regions from top to bottom. Each region: top_pct (0=top of page), height_pct (fraction of page height), and optional doc_type hint.
   Example: passport on top 50%%, medical on bottom 50%% -> regions: [{"top_pct": 0, "height_pct": 0.5, "doc_type": "passport"}, {"top_pct": 0.5, "height_pct": 0.5, "doc_type": "medical_reports"}]
   Use non-overlapping, adjacent bands. Min height_pct 0.08 per region.

4. Categories (BE SPECIFIC):
   - educational_documents: University degrees, diplomas, transcripts, academic certificates
   - experience_certificates: Employment certificates, experience letters, service certificates
   - navttc_reports: NAVTTC vocational training, government technical training
   - police_character_certificate: Police clearance, character certificates
   - certificates: Professional/IT certifications ONLY (CCNA, AWS, PMP, etc.)
   - cv_resume, passport, cnic, driving_license, contracts, medical_reports, photos, other_documents

5. Identity: name, father_name, cnic, passport_no, email, phone, date_of_birth, document_number, nationality, passport_expiry, expiry_date, issue_date, place_of_issue.

Return ONLY valid JSON:
{
  "layout": "single" | "multi",
  "category": "category_name",
  "confidence": 0.95,
  "extracted_identity": { ... },
  "regions": [{"top_pct": 0, "height_pct": 0.5, "doc_type": "passport"}, ...]
}
Omit "regions" when layout is "single". When "multi", "category" and "extracted_identity" apply to the primary/first region; we will re-classify each crop.
"""


def _normalize_identity(raw: dict) -> dict:
    out = {k: raw.get(k) for k in (
        "name", "father_name", "cnic", "passport_no", "email", "phone",
        "date_of_birth", "dob", "document_number", "nationality",
        "passport_expiry", "expiry_date", "issue_date", "place_of_issue",
    )}
    if out.get("expiry_date") and not out.get("passport_expiry"):
        out["passport_expiry"] = out["expiry_date"]
    if out.get("dob") and not out.get("date_of_birth"):
        out["date_of_birth"] = out["dob"]
    if out.get("date_of_birth") and not out.get("dob"):
        out["dob"] = out["date_of_birth"]
    
    # CRITICAL: Validate nationality based on document indicators
    # Pakistani CNIC or Pakistani Passport MUST have nationality = "Pakistani"
    cnic_value = out.get("cnic") or ""
    passport_value = out.get("passport_no") or ""
    place_of_issue = out.get("place_of_issue") or ""
    
    # Check if this is a Pakistani document
    is_pakistani_doc = False
    
    # Check CNIC format (Pakistani CNIC format: 12345-1234567-1 or 13 digits)
    if cnic_value:
        cnic_clean = cnic_value.replace("-", "").replace(" ", "")
        if len(cnic_clean) == 13 and cnic_clean.isdigit():
            is_pakistani_doc = True
    
    # Check passport number (Pakistani passports usually start with PA, AB, or similar)
    if passport_value:
        passport_upper = passport_value.upper().strip()
        if passport_upper.startswith("PA") or passport_upper.startswith("AB"):
            is_pakistani_doc = True
    
    # Check place of issue (Pakistani cities)
    pakistani_cities = ["islamabad", "karachi", "lahore", "peshawar", "quetta", "multan", "faisalabad", "rawalpindi"]
    if place_of_issue.lower() in pakistani_cities:
        is_pakistani_doc = True
    
    # If it's a Pakistani document, enforce nationality
    if is_pakistani_doc:
        if out.get("nationality") and out.get("nationality").lower() != "pakistani":
            logger.warning(f"Overriding nationality from '{out.get('nationality')}' to 'Pakistani' based on document indicators (CNIC/Passport)")
        out["nationality"] = "Pakistani"
    
    return out


def pdf_to_page_images(pdf_bytes: bytes) -> list[tuple[int, bytes]]:
    """Convert PDF to list of (page_idx, png_bytes)."""
    import fitz
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append((i, buf.getvalue()))
    doc.close()
    return result


def image_to_page_images(img_bytes: bytes) -> list[tuple[int, bytes]]:
    """Single image -> one 'page'."""
    return [(0, img_bytes)]


def _parse_vision_json(text: str) -> dict[str, Any]:
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


async def classify_page_vision(image_base64: str, openai_api_key: str) -> dict[str, Any]:
    """Call Vision API for one page; return category, confidence, extracted_identity."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )
    text = resp.choices[0].message.content.strip()
    data = _parse_vision_json(text)
    identity = data.get("extracted_identity") or {}
    data["extracted_identity"] = _normalize_identity(identity)
    return data


async def classify_page_vision_with_layout(image_base64: str, openai_api_key: str) -> dict[str, Any]:
    """Vision layout + classification. Returns layout, category, confidence, identity, regions (if multi)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_LAYOUT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )
    text = resp.choices[0].message.content.strip()
    data = _parse_vision_json(text)
    identity = data.get("extracted_identity") or {}
    data["extracted_identity"] = _normalize_identity(identity)
    data["layout"] = (data.get("layout") or "single").lower()
    data["regions"] = data.get("regions") or []
    return data


def extract_pages_as_pdf_bytes(pdf_bytes: bytes, page_indices: list[int]) -> bytes:
    """Extract given pages from PDF into a new PDF (bytes)."""
    import fitz

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    for i in page_indices:
        dst.insert_pdf(src, from_page=i, to_page=i)
    buf = io.BytesIO()
    dst.save(buf, deflate=True)
    out = buf.getvalue()
    src.close()
    dst.close()
    return out


def image_bytes_to_pdf(img_bytes: bytes) -> bytes:
    """Build 1-page PDF from image bytes (PNG/JPEG)."""
    import fitz
    from PIL import Image

    doc = fitz.open()
    pil = Image.open(io.BytesIO(img_bytes))
    w, h = pil.size
    page = doc.new_page(width=w, height=h)
    img_bio = io.BytesIO()
    pil.save(img_bio, format="PNG")
    # IMPORTANT: PyMuPDF expects raw bytes for `stream=...`, not a BytesIO object.
    # Passing BytesIO can lead to blank pages in some environments.
    page.insert_image(page.rect, stream=img_bio.getvalue())
    buf = io.BytesIO()
    doc.save(buf, deflate=True)
    out = buf.getvalue()
    doc.close()
    return out


def extract_photo_as_jpeg(img_bytes: bytes) -> bytes:
    """
    Convert image to high-quality JPEG for photo documents.
    This ensures photos are stored as actual images, not PDFs.
    Returns JPEG bytes suitable for direct storage and display.
    
    Version: 2.1.0 - Enhanced with embedded image extraction
    """
    from PIL import Image
    
    try:
        # Open and convert to RGB (handles PNG with transparency, etc.)
        pil = Image.open(io.BytesIO(img_bytes))
        
        # Log image info
        logger.info(f"[PhotoExtract] Converting image to JPEG: {pil.size[0]}x{pil.size[1]}px, mode={pil.mode}")
        
        # Handle different image modes
        if pil.mode == 'RGBA':
            # Create white background for transparency
            rgb_image = Image.new('RGB', pil.size, (255, 255, 255))
            rgb_image.paste(pil, mask=pil.split()[3])  # Use alpha channel as mask
            pil = rgb_image
        elif pil.mode != 'RGB':
            pil = pil.convert('RGB')
        
        # Save as high-quality JPEG
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95, optimize=True)
        buf.seek(0)
        
        jpeg_bytes = buf.getvalue()
        logger.info(f"[PhotoExtract] ✅ Created JPEG: {len(jpeg_bytes)} bytes")
        
        return jpeg_bytes
    except Exception as e:
        logger.error(f"[PhotoExtract] ❌ Failed to convert image to JPEG: {e}")
        raise


def extract_photo_from_pdf_page(pdf_bytes: bytes, page_num: int = 0) -> bytes:
    """
    Extract photo from PDF page as JPEG.
    
    Strategy:
    1. Try to extract embedded images first (proper way for photos)
    2. If no images found, convert entire page to JPEG (fallback)
    3. Return the largest image if multiple found
    
    Version: 2.1.0 - Proper embedded image extraction
    Returns: JPEG bytes
    """
    import fitz
    from PIL import Image
    
    logger.info(f"[PhotoExtract] v2.1 - Extracting photo from PDF page {page_num}")
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        
        # Try to extract embedded images first (PROPER WAY)
        image_list = page.get_images(full=True)
        logger.info(f"[PhotoExtract] Found {len(image_list)} embedded images on page")
        
        if image_list:
            # Extract all embedded images
            jpeg_images = []
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    logger.info(f"[PhotoExtract] Image {img_index + 1}: {image_ext}, {len(image_bytes)} bytes")
                    
                    # Convert to JPEG if not already
                    if image_ext.lower() in ['jpg', 'jpeg']:
                        jpeg_bytes = image_bytes
                    else:
                        # Convert using PIL - pass the original image_bytes, not raw pixels!
                        jpeg_bytes = extract_photo_as_jpeg(image_bytes)
                    
                    jpeg_images.append(jpeg_bytes)
                    
                except Exception as e:
                    logger.warning(f"[PhotoExtract] Failed to extract image {img_index + 1}: {e}")
                    continue
            
            if jpeg_images:
                # Return largest image (likely the main photo)
                largest = max(jpeg_images, key=len)
                logger.info(f"[PhotoExtract] ✅ Extracted {len(jpeg_images)} image(s), using largest ({len(largest)} bytes)")
                doc.close()
                return largest
        
        # Fallback: Convert entire page to JPEG
        logger.warning(f"[PhotoExtract] No embedded images found, converting entire page to JPEG")
        
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for quality
        pix = page.get_pixmap(matrix=mat)
        
        logger.info(f"[PhotoExtract] Rendered page: {pix.width}x{pix.height}px")
        
        # Convert pixmap to JPEG via PIL
        img_data = pix.tobytes("ppm")
        pil_image = Image.open(io.BytesIO(img_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=95, optimize=True)
        jpeg_bytes = buf.getvalue()
        
        logger.info(f"[PhotoExtract] ✅ Converted page to JPEG: {len(jpeg_bytes)} bytes")
        
        doc.close()
        pix = None
        
        return jpeg_bytes
        
    except Exception as e:
        logger.error(f"[PhotoExtract] ❌ Failed to extract photo: {e}")
        raise


def apply_confidence_gate(category: str, confidence: float) -> tuple[str, float]:
    """If confidence < threshold, map to other_documents."""
    if confidence < CONFIDENCE_THRESHOLD:
        return "other_documents", confidence
    return category, confidence


def needs_review_from_confidence(confidence: float) -> bool:
    """True when confidence < threshold (ambiguous); no blocking."""
    return confidence < CONFIDENCE_THRESHOLD


def crop_image_by_band(img_bytes: bytes, top_pct: float, height_pct: float) -> bytes:
    """Crop image to vertical band [top_pct, top_pct + height_pct]. Returns PNG bytes."""
    from PIL import Image

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil.size
    y1 = max(0, int(top_pct * h))
    y2 = min(h, int((top_pct + height_pct) * h))
    cropped = pil.crop((0, y1, w, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


MIN_BAND_HEIGHT = 0.08


def group_consecutive_pages(
    documents: list[dict[str, Any]],
    pdf_bytes: bytes | None,
    is_pdf: bool,
) -> list[dict[str, Any]]:
    """
    Phase 2 grouping: consecutive full-page units (split_strategy "page") with same doc_type
    -> one logical document. Rebuild one multi-page PDF per group; set split_strategy "grouped".
    Region units are never grouped; they stay as standalone 1-page docs.
    """
    out: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def flush_group() -> None:
        if not group:
            return
        if len(group) == 1:
            out.append(group[0])
            return
        # Merge 2+ page-level docs from original PDF
        pages_ordered = sorted(set(p for d in group for p in d["pages"]))
        if not is_pdf or not pdf_bytes:
            # Single image or no PDF: cannot merge; emit each as-is (should not reach 2+ with single page)
            for d in group:
                out.append(d)
            return
        merged_pdf = extract_pages_as_pdf_bytes(pdf_bytes, pages_ordered)
        first = group[0]
        conf_min = min(d["confidence"] for d in group)
        any_review = any(d.get("needs_review", False) for d in group)
        out.append({
            "doc_type": first["doc_type"],
            "pages": pages_ordered,
            "regions": [],
            "confidence": conf_min,
            "identity": first.get("identity") or {},
            "pdf_base64": base64.b64encode(merged_pdf).decode("utf-8"),
            "split_strategy": "grouped",
            "needs_review": any_review,
        })

    for doc in documents:
        strategy = doc.get("split_strategy") or "page"
        if strategy == "region":
            flush_group()
            group = []
            out.append(doc)
            continue
        # strategy == "page"
        pages = doc.get("pages") or []
        page_idx = pages[0] if pages else -1
        dtype = doc.get("doc_type") or "other_documents"
        if not group:
            group.append(doc)
            continue
        last_page = max(p for d in group for p in d["pages"]) if group else -2
        same_type = (group[0].get("doc_type") or "other_documents") == dtype
        consecutive = page_idx == last_page + 1
        if same_type and consecutive:
            group.append(doc)
        else:
            flush_group()
            group = [doc]

    flush_group()
    return out


def _append_doc(
    documents: list[dict],
    doc_type: str,
    confidence: float,
    identity: dict,
    pdf_bytes: bytes,
    page_idx: int,
    regions: list,
    split_strategy: str,
    image_bytes: bytes | None = None,  # Optional: raw image bytes for photos
) -> None:
    """
    Append a document to the results list.
    For 'photos' category, saves as JPEG instead of PDF for proper display.
    """
    nr = needs_review_from_confidence(confidence)
    
    # PRODUCTION FIX v2.1: Store photos as JPEG images, not PDFs
    # Extract embedded images from PDF properly
    if doc_type == "photos" and pdf_bytes:
        try:
            # Extract photo from PDF page (proper embedded image extraction)
            logger.info(f"[AppendDoc] Photo document detected (page {page_idx}), extracting as JPEG...")
            jpeg_bytes = extract_photo_from_pdf_page(pdf_bytes, 0)  # Extract from first page of this PDF chunk
            logger.info(f"[AppendDoc] ✅ Photo extracted successfully: {len(jpeg_bytes)} bytes JPEG")
            
            documents.append({
                "doc_type": doc_type,
                "pages": [page_idx],
                "regions": regions,
                "confidence": confidence,
                "identity": identity,
                "pdf_base64": base64.b64encode(jpeg_bytes).decode("utf-8"),
                "split_strategy": split_strategy,
                "needs_review": nr,
                "is_image": True,  # Flag to indicate this is an image, not a PDF
                "mime_type": "image/jpeg",
            })
        except Exception as e:
            logger.error(f"[AppendDoc] ❌ Failed to extract photo as JPEG: {e}, falling back to PDF")
            # Fallback to PDF if JPEG extraction fails
            documents.append({
                "doc_type": doc_type,
                "pages": [page_idx],
                "regions": regions,
                "confidence": confidence,
                "identity": identity,
                "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
                "split_strategy": split_strategy,
                "needs_review": nr,
                "is_image": False,
                "mime_type": "application/pdf",
            })
    else:
        # Standard PDF handling for all other document types
        documents.append({
            "doc_type": doc_type,
            "pages": [page_idx],
            "regions": regions,
            "confidence": confidence,
            "identity": identity,
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
            "split_strategy": split_strategy,
            "needs_review": nr,
            "is_image": False,
            "mime_type": "application/pdf",
        })


async def _process_page_textract_vision(
    page_idx: int,
    img_bytes: bytes,
    pdf_bytes: bytes | None,
    is_pdf: bool,
    aws_region: str,
    openai_api_key: str,
    documents: list[dict],
) -> None:
    """Engine A: Textract layout -> regions; Vision classify. Per-page fallback to Vision on Textract error."""
    from PIL import Image

    pil = Image.open(io.BytesIO(img_bytes))
    w, h = pil.size
    regions: list[dict] = []
    use_tx = False

    if get_blocks_from_image and cluster_blocks_to_regions and validate_no_overlap and crop_image_to_region:
        try:
            blocks = get_blocks_from_image(img_bytes, region=aws_region)
            regions = cluster_blocks_to_regions(blocks, w, h)
            if len(regions) > 1 and validate_no_overlap(regions):
                use_tx = True
            else:
                regions = [{"left": 0, "top": 0, "width": 1.0, "height": 1.0}]
        except Exception as e:
            logger.warning(f"[Split] Textract failed page {page_idx}: {e}, using Vision-only for this page")
            regions = [{"left": 0, "top": 0, "width": 1.0, "height": 1.0}]

    if not use_tx or len(regions) <= 1:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        try:
            v = await classify_page_vision(b64, openai_api_key)
        except Exception as e:
            logger.warning(f"[Split] Vision failed page {page_idx}: {e}")
            v = {"category": "other_documents", "confidence": 0.0, "extracted_identity": {}}
        cat = v.get("category") or "other_documents"
        conf = float(v.get("confidence") or 0.0)
        cat, conf = apply_confidence_gate(cat, conf)
        # Normalize category to canonical doc_type
        doc_type = normalize_doc_type(cat)
        identity = v.get("extracted_identity") or {}
        
        # CRITICAL: Enforce nationality for Pakistani documents
        if doc_type in ['cnic', 'passport']:
            # If document is CNIC or Passport, enforce Pakistani nationality
            if identity.get("nationality") and identity.get("nationality").lower() != "pakistani":
                logger.warning(f"Overriding nationality from '{identity.get('nationality')}' to 'Pakistani' for {doc_type} document")
            identity["nationality"] = "Pakistani"
        
        pdf_one = extract_pages_as_pdf_bytes(pdf_bytes, [page_idx]) if is_pdf and pdf_bytes else image_bytes_to_pdf(img_bytes)
        _append_doc(documents, doc_type, conf, identity, pdf_one, page_idx, [], "page", image_bytes=img_bytes)
        return

    for ri, reg in enumerate(regions):
        try:
            cropped = crop_image_to_region(img_bytes, reg)
        except Exception as e:
            logger.warning(f"[Split] Crop failed page {page_idx} region {ri}: {e}")
            continue
        b64 = base64.b64encode(cropped).decode("utf-8")
        try:
            v = await classify_page_vision(b64, openai_api_key)
        except Exception as e:
            logger.warning(f"[Split] Vision failed page {page_idx} region {ri}: {e}")
            v = {"category": "other_documents", "confidence": 0.0, "extracted_identity": {}}
        cat = v.get("category") or "other_documents"
        conf = float(v.get("confidence") or 0.0)
        cat, conf = apply_confidence_gate(cat, conf)
        # Normalize category to canonical doc_type
        doc_type = normalize_doc_type(cat)
        identity = v.get("extracted_identity") or {}
        
        # CRITICAL: Enforce nationality for Pakistani documents
        if doc_type in ['cnic', 'passport']:
            # If document is CNIC or Passport, enforce Pakistani nationality
            if identity.get("nationality") and identity.get("nationality").lower() != "pakistani":
                logger.warning(f"Overriding nationality from '{identity.get('nationality')}' to 'Pakistani' for {doc_type} document (region)")
            identity["nationality"] = "Pakistani"
        
        pdf_one = image_bytes_to_pdf(cropped)
        _append_doc(documents, doc_type, conf, identity, pdf_one, page_idx, [reg], "region", image_bytes=cropped)


async def _process_page_vision_only(
    page_idx: int,
    img_bytes: bytes,
    pdf_bytes: bytes | None,
    is_pdf: bool,
    openai_api_key: str,
    documents: list[dict],
) -> None:
    """Engine B: Vision layout + classification; optional multi-region bands + 2nd Vision per crop."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    try:
        v = await classify_page_vision_with_layout(b64, openai_api_key)
    except Exception as e:
        logger.warning(f"[Split] Vision layout failed page {page_idx}: {e}")
        v = {"layout": "single", "category": "other_documents", "confidence": 0.0, "extracted_identity": {}, "regions": []}

    layout = (v.get("layout") or "single").lower()
    raw_regions = v.get("regions") or []
    cat = v.get("category") or "other_documents"
    conf = float(v.get("confidence") or 0.0)
    identity = v.get("extracted_identity") or {}

    if layout != "multi" or len(raw_regions) < 2:
        cat, conf = apply_confidence_gate(cat, conf)
        # Normalize category to canonical doc_type
        doc_type = normalize_doc_type(cat)
        
        # CRITICAL: Enforce nationality for Pakistani documents
        if doc_type in ['cnic', 'passport']:
            # If document is CNIC or Passport, enforce Pakistani nationality
            if identity.get("nationality") and identity.get("nationality").lower() != "pakistani":
                logger.warning(f"Overriding nationality from '{identity.get('nationality')}' to 'Pakistani' for {doc_type} document (vision-only single)")
            identity["nationality"] = "Pakistani"
        
        pdf_one = extract_pages_as_pdf_bytes(pdf_bytes, [page_idx]) if is_pdf and pdf_bytes else image_bytes_to_pdf(img_bytes)
        _append_doc(documents, doc_type, conf, identity, pdf_one, page_idx, [], "page", image_bytes=img_bytes)
        return

    # Multi-region: validate bands (min height, non-overlapping), crop each, optional 2nd Vision pass
    bands = []
    for r in raw_regions:
        top = float(r.get("top_pct") or 0)
        height = float(r.get("height_pct") or 0)
        if height < MIN_BAND_HEIGHT:
            continue
        bands.append({"top_pct": top, "height_pct": height, "doc_type_hint": r.get("doc_type")})

    if not bands:
        cat, conf = apply_confidence_gate(cat, conf)
        # Normalize category to canonical doc_type
        doc_type = normalize_doc_type(cat)
        
        # CRITICAL: Enforce nationality for Pakistani documents
        if doc_type in ['cnic', 'passport']:
            # If document is CNIC or Passport, enforce Pakistani nationality
            if identity.get("nationality") and identity.get("nationality").lower() != "pakistani":
                logger.warning(f"Overriding nationality from '{identity.get('nationality')}' to 'Pakistani' for {doc_type} document (vision-only fallback)")
            identity["nationality"] = "Pakistani"
        
        pdf_one = extract_pages_as_pdf_bytes(pdf_bytes, [page_idx]) if is_pdf and pdf_bytes else image_bytes_to_pdf(img_bytes)
        _append_doc(documents, doc_type, conf, identity, pdf_one, page_idx, [], "page", image_bytes=img_bytes)
        return

    for band in bands:
        try:
            cropped = crop_image_by_band(img_bytes, band["top_pct"], band["height_pct"])
        except Exception as e:
            logger.warning(f"[Split] Band crop failed page {page_idx}: {e}")
            continue
        cb64 = base64.b64encode(cropped).decode("utf-8")
        try:
            v2 = await classify_page_vision(cb64, openai_api_key)
        except Exception as e:
            logger.warning(f"[Split] Vision per-region failed page {page_idx}: {e}")
            v2 = {"category": band.get("doc_type_hint") or "other_documents", "confidence": 0.0, "extracted_identity": {}}
        rc = v2.get("category") or band.get("doc_type_hint") or "other_documents"
        rconf = float(v2.get("confidence") or 0.0)
        rc, rconf = apply_confidence_gate(rc, rconf)
        # Normalize category to canonical doc_type
        rdoc_type = normalize_doc_type(rc)
        ridentity = v2.get("extracted_identity") or {}
        
        # CRITICAL: Enforce nationality for Pakistani documents
        if rdoc_type in ['cnic', 'passport']:
            # If document is CNIC or Passport, enforce Pakistani nationality
            if ridentity.get("nationality") and ridentity.get("nationality").lower() != "pakistani":
                logger.warning(f"Overriding nationality from '{ridentity.get('nationality')}' to 'Pakistani' for {rdoc_type} document (multi-region)")
            ridentity["nationality"] = "Pakistani"
        
        pdf_one = image_bytes_to_pdf(cropped)
        reg = {"left": 0, "top": band["top_pct"], "width": 1.0, "height": band["height_pct"]}
        _append_doc(documents, rdoc_type, rconf, ridentity, pdf_one, page_idx, [reg], "region", image_bytes=cropped)


async def run_split_and_categorize(
    file_content: bytes,
    file_name: str,
    is_pdf: bool,
    openai_api_key: str,
    candidate_data: Optional[dict] = None,
    use_textract: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Dual OCR + Vision fallback. Detect engine once (Textract probe); never fail due to Textract.
    Returns { success, engine_used, documents[] } with doc_type, pages, split_strategy, confidence, identity, pdf_base64, needs_review.
    """
    if is_pdf:
        pages = pdf_to_page_images(file_content)
        pdf_bytes = file_content
    else:
        pages = image_to_page_images(file_content)
        pdf_bytes = None

    aws_region = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    documents: list[dict[str, Any]] = []

    engine_used = "vision_only"
    if pages and aws_configured() and detect_engine is not None and (use_textract is None or use_textract):
        first_img = pages[0][1]
        engine_used = detect_engine(first_img, region=aws_region)

    for page_idx, img_bytes in pages:
        if engine_used == "textract+vision":
            await _process_page_textract_vision(
                page_idx, img_bytes, pdf_bytes, is_pdf, aws_region, openai_api_key, documents
            )
        else:
            await _process_page_vision_only(
                page_idx, img_bytes, pdf_bytes, is_pdf, openai_api_key, documents
            )

    # Heuristic: if the upload is mostly a passport, treat nearby low-confidence other_documents pages
    # as passport too (common when users scan blank passport pages).
    try:
        page_docs = [d for d in documents if (d.get("split_strategy") or "page") == "page"]
        passport_docs = [d for d in page_docs if d.get("doc_type") == "passport"]
        if page_docs and passport_docs:
            passport_ratio = len(passport_docs) / max(1, len(page_docs))
            if passport_ratio >= 0.5:
                passport_pages = set()
                for d in passport_docs:
                    for p in (d.get("pages") or []):
                        passport_pages.add(int(p))

                for d in page_docs:
                    if d.get("doc_type") != "other_documents":
                        continue
                    conf = float(d.get("confidence") or 0.0)
                    if conf >= 0.6:
                        continue
                    pages_list = d.get("pages") or []
                    if not pages_list:
                        continue
                    p0 = int(pages_list[0])
                    if any(abs(p0 - pp) <= 2 for pp in passport_pages):
                        d["doc_type"] = "passport"
                        d["needs_review"] = True
    except Exception as e:
        logger.warning(f"[Split] Passport page promotion heuristic failed: {e}")

    # Phase 2: group consecutive full-page units (same doc_type) -> one PDF per group, split_strategy "grouped"
    documents = group_consecutive_pages(documents, pdf_bytes, is_pdf)

    return {"success": True, "engine_used": engine_used, "documents": documents}
