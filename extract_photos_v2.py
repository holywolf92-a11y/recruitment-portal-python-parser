"""
Enhanced Photo Extraction Module
Extracts embedded images from PDFs and converts to JPEG
Version: 2.1.0
"""

import io
import logging
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def extract_embedded_images_from_pdf_page(pdf_bytes: bytes, page_num: int) -> list[bytes]:
    """
    Extract all embedded images from a specific PDF page.
    Returns list of JPEG bytes for each image found.
    
    This is the PROPER way to extract photos from PDFs - extract the embedded
    images directly, not convert the entire page to an image.
    """
    images_found = []
    
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
                
                logger.info(f"[PhotoExtract] Image {img_index + 1}: format={image_ext}, size={len(image_bytes)} bytes")
                
                # Convert to JPEG if not already
                if image_ext.lower() in ['jpg', 'jpeg']:
                    # Already JPEG, use as-is
                    jpeg_bytes = image_bytes
                else:
                    # Convert to JPEG
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert RGBA to RGB if needed
                    if pil_image.mode == 'RGBA':
                        # Create white background
                        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                        rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
                        pil_image = rgb_image
                    elif pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Save as JPEG
                    jpeg_buffer = io.BytesIO()
                    pil_image.save(jpeg_buffer, format='JPEG', quality=95, optimize=True)
                    jpeg_bytes = jpeg_buffer.getvalue()
                    
                    logger.info(f"[PhotoExtract] Converted {image_ext} to JPEG: {len(jpeg_bytes)} bytes")
                
                images_found.append(jpeg_bytes)
                
            except Exception as e:
                logger.error(f"[PhotoExtract] Failed to extract image {img_index + 1}: {e}")
                continue
        
        doc.close()
        
    except Exception as e:
        logger.error(f"[PhotoExtract] Failed to process PDF page {page_num}: {e}")
    
    return images_found


def convert_pdf_page_to_jpeg(pdf_bytes: bytes, page_num: int) -> bytes:
    """
    Fallback: Convert entire PDF page to JPEG image.
    Use this when no embedded images are found (rare for photos).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num]
        
        # Render page at high DPI for quality
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale = 144 DPI
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
