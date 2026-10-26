# Using python 3.9 which often has better pre-built wheel support for dlib
FROM python:3.9-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for opencv-python-headless (libGL + glib).
# Robust against Debian bullseye repo drift that broke the old step:
#  - --allow-releaseinfo-change: tolerate the suite's Release metadata moving
#  - Acquire::Retries + --fix-missing: survive transient mirror fetch failures
#  - libgl1 instead of the transitional libgl1-mesa-glx (always resolvable)
RUN apt-get update -o Acquire::Retries=5 --allow-releaseinfo-change \
    && apt-get install -y --no-install-recommends --fix-missing \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
