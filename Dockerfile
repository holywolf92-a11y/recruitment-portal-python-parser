# Dockerfile for Railway
FROM python:3.11-slim

# Install system dependencies required for dlib and face-recognition
# cmake, build-essential: required to compile dlib
# libx11-6, libxext6, libxrender1: X11 libraries required by face_recognition/dlib
# libgl1-mesa-glx: OpenGL library sometimes required
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
