# Information: Production Docker containerization configuration.
# Importance: Installs required system dependencies (OpenCV) and runs Uvicorn workers behind Gunicorn on Render.

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /workspace

# Install system dependencies (OpenCV headless requires basic libGL/libgthread dependencies occasionally)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY app/ ./app/
COPY models/ ./models/

# Create necessary directories for local storage (ignored by git, needed at runtime)
RUN mkdir -p uploads results logs

# Expose port
EXPOSE 10000

# Command to run FastAPI server
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
