# Dockerfile for Railway
# Using full bullseye image for better compatibility than slim
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# cmake, build-essential: required to compile dlib
# libx11-dev, etc: X11 headers required for dlib GUI support checks
# libopenblas-dev: Linear algebra libraries for dlib optimization
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libx11-dev \
    libxext-dev \
    libsm-dev \
    libxrender-dev \
    libgl1-mesa-glx \
    libopenblas-dev \
    liblapack-dev \
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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
