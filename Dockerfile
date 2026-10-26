# Using python 3.9 which often has better pre-built wheel support for dlib
FROM python:3.9-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# No apt-get step: Debian bullseye is EOL and its security .debs were moved to
# archive.debian.org, so `apt-get install` now fails with exit 100. Nothing here
# needs system packages (all deps ship manylinux wheels; cv2 is not used).

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to minimize image size
# Installing dlib explicitly first can help debug if it fails
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Fail the build (instead of the running service) if the app can't import/start.
# Dummy secrets are inline so they never persist in the image.
RUN PYTHON_HMAC_SECRET=build-smoke OPENAI_API_KEY=build-smoke python -c "import main; from fastapi.testclient import TestClient; r = TestClient(main.app).get('/health'); assert r.status_code == 200, r.text; print('BUILD SMOKE OK')" \
    && pip check

# Expose the port
EXPOSE 8000

# Command to run the application (exec so uvicorn receives SIGTERM)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
