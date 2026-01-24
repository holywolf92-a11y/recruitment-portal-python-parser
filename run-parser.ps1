# Run Python parser with PYTHON_HMAC_SECRET from project (matches backend).
# OPENAI_API_KEY: .env here or "Recruitment Automation Portal (2)/python-parser/.env" fallback.
# AWS (optional): set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION for Textract layout.
$env:PYTHON_HMAC_SECRET = "Itbfr/p8ky/dRMAHLdi/DIiQRLEJtm2SqyNfwuXa3r0="
Set-Location $PSScriptRoot
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
