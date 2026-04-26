"""
Exhaustive split tests for run_split_and_categorize().

Run: python -m tests.test_exhaustive_split
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import fitz  # noqa: E402
import split_and_categorize as sac  # noqa: E402


def _doc(doc_type: str, page_idx: int, confidence: float = 0.95) -> dict:
    return {
        "doc_type": doc_type,
        "pages": [page_idx],
        "regions": [],
        "confidence": confidence,
        "identity": {},
        "pdf_base64": base64.b64encode(f"page-{page_idx}".encode("utf-8")).decode("utf-8"),
        "split_strategy": "page",
        "needs_review": False,
    }


def _make_pdf(n_pages: int) -> bytes:
        doc = fitz.open()
        for i in range(n_pages):
            page = doc.new_page(width=100, height=100)
            page.insert_text((10, 50), str(i), fontsize=20)
        buf = io.BytesIO()
        doc.save(buf, deflate=True)
        out = buf.getvalue()
        doc.close()
        return out


async def test_exhaustive_page_scan_disables_grouping() -> None:
    original_pdf_to_page_images = sac.pdf_to_page_images
    original_process_page = sac._process_page_vision_only

    try:
        sac.pdf_to_page_images = lambda _: [(0, b"p0"), (1, b"p1"), (2, b"p2")]  # type: ignore[assignment]

        async def fake_process(page_idx, img_bytes, pdf_bytes, is_pdf, openai_api_key, documents):
            doc_types = ["passport", "passport", "cv_resume"]
            documents.append(_doc(doc_types[page_idx], page_idx))

        sac._process_page_vision_only = fake_process  # type: ignore[assignment]

        result = await sac.run_split_and_categorize(
            file_content=b"pdf",
            file_name="bundle.pdf",
            is_pdf=True,
            openai_api_key="test",
            exhaustive_page_scan=True,
        )

        assert [doc["doc_type"] for doc in result["documents"]] == ["passport", "passport", "cv_resume"]
        assert all(doc["split_strategy"] == "page" for doc in result["documents"])
    finally:
        sac.pdf_to_page_images = original_pdf_to_page_images  # type: ignore[assignment]
        sac._process_page_vision_only = original_process_page  # type: ignore[assignment]


async def test_default_mode_can_collapse_to_single_group() -> None:
    original_pdf_to_page_images = sac.pdf_to_page_images
    original_process_page = sac._process_page_vision_only

    try:
        sac.pdf_to_page_images = lambda _: [(0, b"p0"), (1, b"p1"), (2, b"p2")]  # type: ignore[assignment]

        async def fake_process(page_idx, img_bytes, pdf_bytes, is_pdf, openai_api_key, documents):
            doc_types = ["passport", "passport", "cv_resume"]
            documents.append(_doc(doc_types[page_idx], page_idx))

        sac._process_page_vision_only = fake_process  # type: ignore[assignment]

        result = await sac.run_split_and_categorize(
            file_content=_make_pdf(3),
            file_name="bundle.pdf",
            is_pdf=True,
            openai_api_key="test",
            exhaustive_page_scan=False,
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0]["doc_type"] == "passport"
        assert result["documents"][0]["pages"] == [0, 1, 2]
        assert result["documents"][0]["split_strategy"] == "grouped"
    finally:
        sac.pdf_to_page_images = original_pdf_to_page_images  # type: ignore[assignment]
        sac._process_page_vision_only = original_process_page  # type: ignore[assignment]


def run_all() -> None:
    asyncio.run(test_exhaustive_page_scan_disables_grouping())
    print("  test_exhaustive_page_scan_disables_grouping: OK")
    asyncio.run(test_default_mode_can_collapse_to_single_group())
    print("  test_default_mode_can_collapse_to_single_group: OK")
    print("All exhaustive split tests passed.")


if __name__ == "__main__":
    run_all()