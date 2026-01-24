#!/usr/bin/env python3
"""
Enable IAM permissions required for the document-split pipeline (AWS Textract).

Usage:
  python enable_textract_permissions.py [--user IAM_USERNAME] [--verify-only] [--region REGION]

- Run with **admin** credentials (root or IAM user with IAM access).
- --user: IAM user to attach the policy to (e.g. textract-service). Required if caller is root.
- --verify-only: Skip policy creation; just verify AWS creds + Textract access (e.g. parser user).
- --region: AWS region for Textract (default: us-east-1).

Loads AWS_* from .env (parser dir or "Recruitment Automation Portal (2)/python-parser/.env").
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

# Load .env before importing boto3 (so AWS_* are set)
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    fallback = root.parent / "Recruitment Automation Portal (2)" / "python-parser" / ".env"
    if fallback.exists():
        load_dotenv(fallback)


_load_dotenv()

POLICY_NAME = "RecruitmentPortalTextractPolicy"
POLICY_DOC = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "TextractDetectDocumentText",
            "Effect": "Allow",
            "Action": ["textract:DetectDocumentText"],
            "Resource": "*",
        }
    ],
}


def get_caller(boto_sts):
    resp = boto_sts.get_caller_identity()
    arn = resp.get("Arn") or ""
    account = resp.get("Account") or ""
    return arn, account


def is_root(arn: str) -> bool:
    return ":root" in arn or arn.endswith("/root")


def iam_user_from_arn(arn: str) -> str | None:
    if "/user/" in arn:
        return arn.split("/user/")[-1].split("/")[0]
    return None


def create_policy(iam, account: str):
    """Create or get existing policy. Returns policy ARN."""
    policy_arn = f"arn:aws:iam::{account}:policy/{POLICY_NAME}"
    try:
        iam.create_policy(
            PolicyName=POLICY_NAME,
            PolicyDocument=json.dumps(POLICY_DOC),
            Description="Minimal Textract access for document-split pipeline (DetectDocumentText only).",
        )
        print(f"Created IAM policy: {POLICY_NAME}")
        return policy_arn
    except iam.exceptions.EntityAlreadyExistsException:
        try:
            iam.create_policy_version(
                PolicyArn=policy_arn,
                PolicyDocument=json.dumps(POLICY_DOC),
                SetAsDefault=True,
            )
            print(f"Updated existing policy: {POLICY_NAME}")
        except iam.exceptions.LimitExceededException:
            print(f"Policy {POLICY_NAME} exists (max versions); using current.")
        except Exception as e:
            raise
        return policy_arn


def attach_policy_to_user(iam, policy_arn: str, user: str) -> None:
    iam.attach_user_policy(UserName=user, PolicyArn=policy_arn)
    print(f"Attached {POLICY_NAME} to IAM user: {user}")


def verify_textract(region: str) -> None:
    """Call DetectDocumentText with a minimal image to verify permissions."""
    import boto3
    from PIL import Image

    client = boto3.client("textract", region_name=region)
    img = Image.new("RGB", (50, 50), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    doc_bytes = buf.read()
    try:
        client.detect_document_text(Document={"Bytes": doc_bytes})
    except client.exceptions.AccessDeniedException as e:
        raise RuntimeError(
            f"Textract access denied. Attach {POLICY_NAME} to your IAM user. {e}"
        ) from e
    except Exception as e:
        err = str(e).lower()
        if "subscriptionrequired" in err or "subscription" in err:
            raise RuntimeError(
                "Textract not enabled in this account/region. "
                "Enable it: AWS Console -> Textract -> Get started (first-time use). "
                f"Details: {e}"
            ) from e
        raise
    print("Textract DetectDocumentText: OK (permissions verified).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Enable Textract IAM permissions for document-split pipeline.")
    ap.add_argument("--user", metavar="IAM_USERNAME", help="IAM user to attach policy to (e.g. textract-service). Required if caller is root.")
    ap.add_argument("--verify-only", action="store_true", help="Only verify AWS + Textract access; skip policy create/attach.")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION") or "us-east-1", help="AWS region for Textract.")
    ap.add_argument("--try-regions", action="store_true", help="With --verify-only: try us-east-1, us-east-2, eu-west-1, ap-south-1 and report which works.")
    args = ap.parse_args()

    try:
        import boto3
    except ImportError:
        print("boto3 is required. Run: pip install boto3")
        return 1

    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (env or .env).")
        return 1

    sts = boto3.client("sts")
    iam = boto3.client("iam")

    try:
        arn, account = get_caller(sts)
    except Exception as e:
        print(f"AWS credentials invalid: {e}")
        return 1

    print(f"Caller: {arn}")
    print(f"Account: {account}")

    if args.verify_only:
        regions_to_try = (
            ["us-east-1", "us-east-2", "eu-west-1", "ap-south-1"]
            if args.try_regions
            else [args.region]
        )
        if args.try_regions:
            print(f"Trying regions: {', '.join(regions_to_try)}")
        else:
            print(f"Region: {args.region}")
        last_err = None
        for r in regions_to_try:
            try:
                verify_textract(r)
                if args.try_regions:
                    print(f"  OK in {r}")
                return 0
            except Exception as e:
                last_err = e
                if args.try_regions:
                    print(f"  {r}: {e}")
        print(f"Textract verify failed: {last_err}")
        err_str = str(last_err).lower()
        if "subscriptionrequired" in err_str or "subscription" in err_str:
            print("")
            print("Next steps:")
            print("  1. Open AWS Console -> Textract (same region as AWS_DEFAULT_REGION).")
            print("     https://console.aws.amazon.com/textract/")
            print("  2. Click 'Get started' / run a demo if first-time use.")
            print("  3. If using *root* keys: try IAM user keys (e.g. textract-service) in .env instead.")
            if is_root(arn):
                print("     (You are using root. Use textract-service access keys + run --verify-only again.)")
        return 1

    root = is_root(arn)
    self_user = iam_user_from_arn(arn)

    target_user = args.user
    if root and not target_user:
        print("Caller is root. Pass --user IAM_USERNAME (e.g. textract-service) to attach the policy.")
        return 1
    if not target_user:
        target_user = self_user
    if not target_user:
        print("Could not determine IAM user from ARN. Pass --user IAM_USERNAME.")
        return 1

    try:
        policy_arn = create_policy(iam, account)
        attach_policy_to_user(iam, policy_arn, target_user)
    except Exception as e:
        print(f"Failed: {e}")
        return 1

    print("")
    print("Done. The IAM user has textract:DetectDocumentText. Use that user's keys for the parser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
