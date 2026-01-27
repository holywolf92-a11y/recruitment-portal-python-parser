#!/usr/bin/env python3
"""
Check if AWS (Textract) is configured and working.
Uses same logic as split_and_categorize: aws_configured() + optional Textract probe.
Loads .env from parser dir (or fallback) so keys you added there are used.
Run from parser dir: python check_aws.py
"""
import os
import sys
from pathlib import Path

def _load_dotenv():
    """Load .env from parser dir, then fallback (same as main.py / enable_textract_permissions)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return []
    root = Path(__file__).resolve().parent
    env1 = root / ".env"
    fallback = root.parent / "Recruitment Automation Portal (2)" / "python-parser" / ".env"
    load_dotenv(env1)
    if fallback.exists():
        load_dotenv(fallback)
    return [env1, fallback]

def main():
    print("=== AWS / Textract check ===\n")

    # 0) Load .env from parser dir or fallback (where you may have put AWS keys)
    tried = _load_dotenv()
    if tried:
        print("Loaded .env from:")
        for p in tried:
            print(f"  {p} (exists: {p.exists()})")
        print()

    # 0b) Map common alternate names -> AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    if not os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY"]
    if not os.environ.get("AWS_SECRET_ACCESS_KEY") and os.environ.get("AWS_SECRET_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_KEY"]

    # 1) Env vars
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

    print("Environment:")
    print(f"  AWS_ACCESS_KEY_ID:    {'(set)' if ak else '(not set)'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'(set)' if sk else '(not set)'}")
    print(f"  AWS_DEFAULT_REGION:   {region}")
    print()

    if not (ak and sk):
        print("Result: AWS not configured (missing credentials).")
        print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to use Textract.")
        print()
        print("Where to put them:")
        print("  1. recruitment-portal-python-parser\\.env (create it) with exact lines:")
        print("     AWS_ACCESS_KEY_ID=your-access-key-id")
        print("     AWS_SECRET_ACCESS_KEY=your-secret-access-key")
        print("     AWS_DEFAULT_REGION=us-east-1")
        print("  2. Or in \"Recruitment Automation Portal (2)\\python-parser\\.env\" using the same names.")
        print("  3. Or set in shell: $env:AWS_ACCESS_KEY_ID=\"AKIA...\"; $env:AWS_SECRET_ACCESS_KEY=\"...\"")
        print()
        print("Parser will use vision-only mode until those are set. See AWS_SETUP_GUIDE.md.")
        return 1

    # 2) Textract probe (same as textract_layout.detect_engine)
    try:
        from textract_layout import aws_configured, detect_engine
    except ImportError:
        print("Result: textract_layout not found (wrong cwd?). Run from recruitment-portal-python-parser/")
        return 2

    if not aws_configured():
        print("Result: aws_configured() is False (unexpected after env check).")
        return 1

    # Minimal 1x1 PNG (valid image bytes for API)
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
        b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    print("Textract probe (detect_document_text on minimal image)...")
    engine = detect_engine(minimal_png, region=region)

    if engine == "textract+vision":
        print("Result: AWS Textract is working. Parser will use Textract+Vision.")
        return 0
    else:
        print("Result: Textract probe returned vision_only (credentials/perms/network issue).")
        print("Parser will still run in vision-only mode.")
        return 0  # not a hard failure; app continues with Vision

if __name__ == "__main__":
    sys.exit(main() or 0)
