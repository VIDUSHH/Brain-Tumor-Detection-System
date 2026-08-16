# Hugging Face Spaces Docker SDK
# Free tier: 2 vCPU / 16 GB RAM / 50 GB disk. Listens on port 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /workspace

# OpenCV headless + TensorFlow runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/best_model.h5 ./models/best_model.h5

RUN mkdir -p uploads results logs models

EXPOSE 7860

# 2 workers for the 2 vCPU free instance; 120s timeout for slow first inference
CMD gunicorn -w 2 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT --timeout 120
