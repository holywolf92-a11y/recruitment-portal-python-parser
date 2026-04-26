"""
Enhanced Photo Extraction Module
Extracts embedded images from PDFs and provides reusable crop and normalize helpers.
Version: 2.2.0
"""

from dataclasses import dataclass
import io
import logging
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PhotoCandidate:
    source: str
    page_num: int
    image_bytes: bytes
    width: int
    height: int
    image_index: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None


def _image_bytes_to_jpeg(image_bytes: bytes) -> tuple[bytes, int, int]:
    pil_image = Image.open(io.BytesIO(image_bytes))

    if pil_image.mode == 'RGBA':
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        rgb_image.paste(pil_image, mask=pil_image.split()[3])
        pil_image = rgb_image
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    jpeg_buffer = io.BytesIO()
    pil_image.save(jpeg_buffer, format='JPEG', quality=95, optimize=True)
    return jpeg_buffer.getvalue(), pil_image.width, pil_image.height


def extract_embedded_image_candidates_from_pdf_page(pdf_bytes: bytes, page_num: int) -> list[PhotoCandidate]:
    """
    Extract all embedded images from a specific PDF page.
    Returns image candidates with normalized JPEG bytes and dimensions.
    
    This is the PROPER way to extract photos from PDFs - extract the embedded
    images directly, not convert the entire page to an image.
    """
    images_found: list[PhotoCandidate] = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        
        # Get all images on this page
        image_list = page.get_images(full=True)
        
        logger.info(f"[PhotoExtract] Found {len(image_list)} embedded images on page {page_num}")
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # xref is the image reference number
            
            try:
                # Extract the image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # Original format (png, jpg, etc.)
                img_w = int(base_image.get("width", 0) or 0)
                img_h = int(base_image.get("height", 0) or 0)
                
                logger.info(f"[PhotoExtract] Image {img_index + 1}: format={image_ext}, size={len(image_bytes)} bytes")
                jpeg_bytes, converted_w, converted_h = _image_bytes_to_jpeg(image_bytes)

                if image_ext.lower() not in ['jpg', 'jpeg']:
                    logger.info(f"[PhotoExtract] Converted {image_ext} to JPEG: {len(jpeg_bytes)} bytes")

                images_found.append(PhotoCandidate(
                    source='embedded',
                    page_num=page_num,
                    image_bytes=jpeg_bytes,
                    width=img_w or converted_w,
                    height=img_h or converted_h,
                    image_index=img_index,
                ))
                
            except Exception as e:
                logger.error(f"[PhotoExtract] Failed to extract image {img_index + 1}: {e}")
                continue
        
        doc.close()
        
    except Exception as e:
        logger.error(f"[PhotoExtract] Failed to process PDF page {page_num}: {e}")
    
    return images_found


def extract_displayed_image_candidates_from_pdf_page(pdf_bytes: bytes, page_num: int) -> list[PhotoCandidate]:
    """
    Extract image blocks actually displayed on a page, including their page bbox.

    This uses Page.get_text("dict") because it returns the rendered image blocks
    with both binary image data and page coordinates. That makes it suitable for
    ranking likely CV avatar/headshot images before falling back to full-page scans.
    """
    images_found: list[PhotoCandidate] = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        page_dict = page.get_text("dict")
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        blocks = page_dict.get("blocks") or []
        image_blocks = [block for block in blocks if block.get("type") == 1 and block.get("image")]
        logger.info(f"[PhotoExtract] Found {len(image_blocks)} displayed image blocks on page {page_num}")

        for img_index, block in enumerate(image_blocks):
            try:
                image_bytes = block["image"]
                jpeg_bytes, converted_w, converted_h = _image_bytes_to_jpeg(image_bytes)
                bbox = tuple(block.get("bbox") or ())
                if len(bbox) != 4:
                    bbox = None

                images_found.append(PhotoCandidate(
                    source='displayed_image_block',
                    page_num=page_num,
                    image_bytes=jpeg_bytes,
                    width=int(block.get("width", 0) or converted_w),
                    height=int(block.get("height", 0) or converted_h),
                    image_index=img_index,
                    bbox=bbox,
                    page_width=page_width,
                    page_height=page_height,
                ))
            except Exception as e:
                logger.error(f"[PhotoExtract] Failed to process displayed image block {img_index + 1}: {e}")
                continue

        doc.close()
    except Exception as e:
        logger.error(f"[PhotoExtract] Failed to extract displayed image blocks from page {page_num}: {e}")

    return images_found


def extract_embedded_images_from_pdf_page(pdf_bytes: bytes, page_num: int) -> list[bytes]:
    """Backward-compatible wrapper that returns only JPEG bytes."""
    return [candidate.image_bytes for candidate in extract_embedded_image_candidates_from_pdf_page(pdf_bytes, page_num)]


def render_pdf_page_to_jpeg(pdf_bytes: bytes, page_num: int, scale: float = 2.0) -> bytes:
    """
    Render a PDF page to a JPEG image for fallback face detection.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        
        # Render page at high DPI for quality
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        
        logger.info(f"[PhotoExtract] Rendered page {page_num} as pixmap: {pix.width}x{pix.height}px")
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("ppm")  # PPM format for PIL
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Save as JPEG
        jpeg_buffer = io.BytesIO()
        pil_image.save(jpeg_buffer, format='JPEG', quality=95, optimize=True)
        jpeg_bytes = jpeg_buffer.getvalue()
        
        logger.info(f"[PhotoExtract] Converted page to JPEG: {len(jpeg_bytes)} bytes")
        
        doc.close()
        pix = None  # Free memory
        
        return jpeg_bytes
        
    except Exception as e:
        logger.error(f"[PhotoExtract] Failed to convert page {page_num} to JPEG: {e}")
        raise


def convert_pdf_page_to_jpeg(pdf_bytes: bytes, page_num: int) -> bytes:
    """Backward-compatible wrapper."""
    return render_pdf_page_to_jpeg(pdf_bytes, page_num, scale=2.0)


def crop_image_with_padding(
    image_bytes: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    padding_ratio: float = 0.30,
) -> Image.Image:
    """Crop an image box with padding and normalize to RGB."""
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    width, height = pil_image.size
    face_w = max(1, right - left)
    face_h = max(1, bottom - top)
    padding = int(max(face_w, face_h) * padding_ratio)

    x1 = max(0, left - padding)
    y1 = max(0, top - padding)
    x2 = min(width, right + padding)
    y2 = min(height, bottom + padding)
    return pil_image.crop((x1, y1, x2, y2))


def normalize_face_crop_to_jpeg(face_crop: Image.Image, target_size: int = 512) -> bytes:
    """Convert a face crop into a square JPEG."""
    if face_crop.mode != 'RGB':
        face_crop = face_crop.convert('RGB')

    crop_w, crop_h = face_crop.size
    if crop_w != crop_h:
        max_dim = max(crop_w, crop_h)
        square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        x_offset = (max_dim - crop_w) // 2
        y_offset = (max_dim - crop_h) // 2
        square_img.paste(face_crop, (x_offset, y_offset))
        face_crop = square_img

    face_crop = face_crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
    jpeg_buffer = io.BytesIO()
    face_crop.save(jpeg_buffer, format='JPEG', quality=95, optimize=True)
    return jpeg_buffer.getvalue()


def extract_photo_from_pdf(pdf_bytes: bytes, page_num: int = 0) -> bytes:
    """
    Main function: Extract photo from PDF page as JPEG.
    
    Strategy:
    1. Try to extract embedded images first (proper way)
    2. If no images found, convert entire page to JPEG (fallback)
    3. Return the largest image if multiple found
    
    Returns: JPEG bytes
    """
    logger.info(f"[PhotoExtract] Starting photo extraction from page {page_num}")
    
    # Try embedded image extraction first
    embedded_images = extract_embedded_images_from_pdf_page(pdf_bytes, page_num)
    
    if embedded_images:
        # Return the largest image (likely the photo)
        largest_image = max(embedded_images, key=len)
        logger.info(f"[PhotoExtract] ✅ Extracted {len(embedded_images)} image(s), using largest ({len(largest_image)} bytes)")
        return largest_image
    
    # Fallback: convert entire page
    logger.warning(f"[PhotoExtract] No embedded images found, converting entire page")
    return convert_pdf_page_to_jpeg(pdf_bytes, page_num)
