"""
AWS Textract layout: blocks -> regions.

Use when AWS credentials are set. Provides precise bounding boxes for
Possibility 3 (multiple docs on same page). Falls back to Vision-only when unavailable.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Gap (as fraction of page height) above which we start a new region
VERTICAL_GAP_THRESHOLD = 0.08
# Min region area (fraction of page) to avoid noise
MIN_REGION_AREA = 0.01
# Min width/height (fraction)
MIN_REGION_DIM = 0.05


def aws_configured() -> bool:
    """True if AWS credentials are available (env vars)."""
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    return bool(ak and sk)


# Textract failure cases that MUST trigger fallback to vision_only (instruction)
TEXTRACT_FALLBACK_ERROR_CODES = {
    "SubscriptionRequiredException",
    "AccessDeniedException",
    "EndpointConnectionError",
    "ProvisionedThroughputExceededException",
    "ThrottlingException",
    "RequestTimeout",
    "ReadTimeout",
    "ConnectTimeout",
}


def detect_engine(img_bytes: bytes, region: str = "us-east-1") -> str:
    """
    Lightweight Textract probe. Returns "textract+vision" if available, else "vision_only".
    Never raises; all Textract failures -> vision_only.
    """
    import boto3

    try:
        client = boto3.client("textract", region_name=region)
        client.detect_document_text(Document={"Bytes": img_bytes})
        return "textract+vision"
    except Exception as e:
        code = ""
        if hasattr(e, "response") and isinstance(getattr(e, "response", None), dict):
            code = (e.response.get("Error") or {}).get("Code") or ""
        code = str(code).strip()
        err_str = str(e).lower()
        if code in TEXTRACT_FALLBACK_ERROR_CODES:
            logger.info(f"[Engine] Textract unavailable ({code}), using vision_only.")
            return "vision_only"
        if "subscriptionrequired" in err_str or "accessdenied" in err_str or "throttl" in err_str:
            logger.info(f"[Engine] Textract unavailable ({e}), using vision_only.")
            return "vision_only"
        if "timeout" in err_str or "connection" in err_str or "endpoint" in err_str:
            logger.info(f"[Engine] Textract connection/timeout ({e}), using vision_only.")
            return "vision_only"
        logger.warning(f"[Engine] Textract error ({e}), using vision_only.")
        return "vision_only"


def get_blocks_from_image(img_bytes: bytes, region: str = "us-east-1") -> list[dict[str, Any]]:
    """
    Call Textract DetectDocumentText on a single page image.
    Returns list of blocks (LINE/WORD) with Geometry.BoundingBox.
    """
    import boto3

    client = boto3.client("textract", region_name=region)
    resp = client.detect_document_text(Document={"Bytes": img_bytes})
    blocks = resp.get("Blocks") or []
    # Use LINE blocks for layout (larger, fewer); fall back to WORD if no LINES
    out = [b for b in blocks if b.get("BlockType") in ("LINE", "WORD")]
    if not out:
        out = [b for b in blocks if b.get("BlockType") == "WORD"]
    return out


def _bbox_area(b: dict) -> float:
    g = b.get("Geometry") or {}
    bb = g.get("BoundingBox") or {}
    w = float(bb.get("Width") or 0)
    h = float(bb.get("Height") or 0)
    return w * h


def _bbox_center_y(b: dict) -> float:
    g = b.get("Geometry") or {}
    bb = g.get("BoundingBox") or {}
    top = float(bb.get("Top") or 0)
    height = float(bb.get("Height") or 0)
    return top + height / 2


def _merge_bboxes(blocks: list[dict]) -> dict[str, float]:
    """Compute union bbox of blocks. Returns {left, top, width, height} normalized 0-1."""
    if not blocks:
        return {"left": 0, "top": 0, "width": 0.01, "height": 0.01}
    g0 = (blocks[0].get("Geometry") or {}).get("BoundingBox") or {}
    l_min = float(g0.get("Left") or 0)
    t_min = float(g0.get("Top") or 0)
    r_max = l_min + float(g0.get("Width") or 0)
    b_max = t_min + float(g0.get("Height") or 0)
    for b in blocks[1:]:
        g = (b.get("Geometry") or {}).get("BoundingBox") or {}
        L = float(g.get("Left") or 0)
        T = float(g.get("Top") or 0)
        W = float(g.get("Width") or 0)
        H = float(g.get("Height") or 0)
        l_min = min(l_min, L)
        t_min = min(t_min, T)
        r_max = max(r_max, L + W)
        b_max = max(b_max, T + H)
    return {
        "left": l_min,
        "top": t_min,
        "width": max(0.01, r_max - l_min),
        "height": max(0.01, b_max - t_min),
    }


def cluster_blocks_to_regions(
    blocks: list[dict],
    page_width: int,
    page_height: int,
    gap_threshold: float = VERTICAL_GAP_THRESHOLD,
    min_area: float = MIN_REGION_AREA,
    min_dim: float = MIN_REGION_DIM,
) -> list[dict[str, float]]:
    """
    Cluster blocks by vertical proximity. Gaps > gap_threshold (fraction of page height)
    start a new region. Returns list of {left, top, width, height} in normalized 0-1.
    """
    if not blocks:
        return [{"left": 0, "top": 0, "width": 1.0, "height": 1.0}]

    sorted_blocks = sorted(blocks, key=_bbox_center_y)
    clusters: list[list[dict]] = []
    current: list[dict] = [sorted_blocks[0]]

    for b in sorted_blocks[1:]:
        g = (b.get("Geometry") or {}).get("BoundingBox") or {}
        this_top = float(g.get("Top") or 0)
        prev = current[-1]
        pg = (prev.get("Geometry") or {}).get("BoundingBox") or {}
        prev_bottom = float(pg.get("Top") or 0) + float(pg.get("Height") or 0)
        gap = this_top - prev_bottom
        if gap > gap_threshold:
            clusters.append(current)
            current = [b]
        else:
            current.append(b)
    if current:
        clusters.append(current)

    regions = []
    for c in clusters:
        box = _merge_bboxes(c)
        a = box["width"] * box["height"]
        if a < min_area or box["width"] < min_dim or box["height"] < min_dim:
            continue
        regions.append(box)

    if not regions:
        return [{"left": 0, "top": 0, "width": 1.0, "height": 1.0}]
    return regions


def iou(a: dict, b: dict) -> float:
    """Intersection over union of two normalized bboxes."""
    ax1, ay1 = a["left"], a["top"]
    ax2, ay2 = a["left"] + a["width"], a["top"] + a["height"]
    bx1, by1 = b["left"], b["top"]
    bx2, by2 = b["left"] + b["width"], b["top"] + b["height"]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_no_overlap(regions: list[dict], epsilon: float = 0.01) -> bool:
    """True if no pair of regions overlaps (IoU < epsilon)."""
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if iou(regions[i], regions[j]) >= epsilon:
                return False
    return True


def crop_image_to_region(img_bytes: bytes, region: dict[str, float]) -> bytes:
    """Crop image to normalized region {left, top, width, height}. Returns PNG bytes."""
    from PIL import Image

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil.size
    x1 = int(region["left"] * w)
    y1 = int(region["top"] * h)
    x2 = min(w, int((region["left"] + region["width"]) * w))
    y2 = min(h, int((region["top"] + region["height"]) * h))
    cropped = pil.crop((max(0, x1), max(0, y1), x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()
