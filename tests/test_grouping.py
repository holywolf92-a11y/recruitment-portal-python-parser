"""
Phase 2 grouping tests for group_consecutive_pages().

Run: python -m tests.test_grouping
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

# Add parser root so we can import split_and_categorize
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import fitz  # noqa: E402

from split_and_categorize import group_consecutive_pages  # noqa: E402


def _make_pdf(n_pages: int) -> bytes:
    doc = fitz.open()
    for i in range(n_pages):
        p = doc.new_page(width=100, height=100)
        p.insert_text((10, 50), str(i), fontsize=20)
    buf = io.BytesIO()
    doc.save(buf, deflate=True)
    out = buf.getvalue()
    doc.close()
    return out


def _page_doc(doc_type: str, page: int, conf: float = 0.9, needs_review: bool = False) -> dict:
    return {
        "doc_type": doc_type,
        "pages": [page],
        "regions": [],
        "confidence": conf,
        "identity": {},
        "pdf_base64": "x",
        "split_strategy": "page",
        "needs_review": needs_review,
    }


def _region_doc(doc_type: str, page: int, conf: float = 0.9) -> dict:
    return {
        "doc_type": doc_type,
        "pages": [page],
        "regions": [{}],
        "confidence": conf,
        "identity": {},
        "pdf_base64": "x",
        "split_strategy": "region",
        "needs_review": False,
    }


def test_three_cv_one_passport() -> None:
    """3 consecutive cv (0,1,2) + 1 passport (3) -> 2 docs: grouped cv, page passport."""
    pdf = _make_pdf(4)
    docs = [_page_doc("cv_resume", i) for i in range(3)] + [_page_doc("passport", 3)]
    r = group_consecutive_pages(docs, pdf, True)
    assert len(r) == 2, r
    assert r[0]["split_strategy"] == "grouped" and r[0]["pages"] == [0, 1, 2]
    assert r[1]["split_strategy"] == "page" and r[1]["pages"] == [3]


def test_region_in_between() -> None:
    """Page cv, region passport, region medical, page cv -> 4 docs; no grouping across regions."""
    pdf = _make_pdf(4)
    a = _page_doc("cv_resume", 0)
    b = _region_doc("passport", 1)
    c = _region_doc("medical_reports", 1)
    d = _page_doc("cv_resume", 2)
    r = group_consecutive_pages([a, b, c, d], pdf, True)
    assert len(r) == 4
    assert r[0]["split_strategy"] == "page" and r[0]["doc_type"] == "cv_resume"
    assert r[1]["split_strategy"] == "region" and r[1]["doc_type"] == "passport"
    assert r[2]["split_strategy"] == "region" and r[2]["doc_type"] == "medical_reports"
    assert r[3]["split_strategy"] == "page" and r[3]["doc_type"] == "cv_resume"


def test_cv_passport_cv() -> None:
    """cv, passport, cv -> 3 separate docs; no grouping (different types in between)."""
    pdf = _make_pdf(3)
    docs = [_page_doc("cv_resume", 0), _page_doc("passport", 1), _page_doc("cv_resume", 2)]
    r = group_consecutive_pages(docs, pdf, True)
    assert len(r) == 3
    assert all(d["split_strategy"] == "page" for d in r)


def test_merged_pdf_min_confidence_needs_review() -> None:
    """Grouped doc: merged PDF has 3 pages; confidence = min; needs_review = any true."""
    pdf = _make_pdf(4)
    a = _page_doc("cv_resume", 0, 0.92)
    b = _page_doc("cv_resume", 1, 0.94)
    c = _page_doc("cv_resume", 2, 0.88, needs_review=True)
    r = group_consecutive_pages([a, b, c], pdf, True)
    assert len(r) == 1
    assert r[0]["split_strategy"] == "grouped"
    assert r[0]["confidence"] == 0.88
    assert r[0]["needs_review"] is True
    merged = fitz.open(stream=base64.b64decode(r[0]["pdf_base64"]), filetype="pdf")
    assert len(merged) == 3
    merged.close()


def test_empty_and_single_no_pdf() -> None:
    """Empty list -> []. Single doc, no PDF -> 1 doc, page."""
    r = group_consecutive_pages([], None, False)
    assert r == []
    d = _page_doc("cv_resume", 0)
    r = group_consecutive_pages([d], None, False)
    assert len(r) == 1 and r[0]["split_strategy"] == "page"


def test_non_consecutive_same_type() -> None:
    """Page 0 cv, page 2 cv (skip 1) -> 2 separate docs; no grouping."""
    pdf = _make_pdf(4)
    docs = [_page_doc("cv_resume", 0), _page_doc("cv_resume", 2)]
    r = group_consecutive_pages(docs, pdf, True)
    assert len(r) == 2
    assert r[0]["pages"] == [0] and r[1]["pages"] == [2]
    assert all(d["split_strategy"] == "page" for d in r)


def run_all() -> None:
    tests = [
        test_three_cv_one_passport,
        test_region_in_between,
        test_cv_passport_cv,
        test_merged_pdf_min_confidence_needs_review,
        test_empty_and_single_no_pdf,
        test_non_consecutive_same_type,
    ]
    for t in tests:
        t()
        print(f"  {t.__name__}: OK")
    print("All grouping tests passed.")


if __name__ == "__main__":
    run_all()
