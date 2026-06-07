# Information: Main application entry point for the FastAPI server.
# Importance: Registers routers, sets up CORS middleware, mounts results folder for static access, and configures rotating file loggers.

import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf

from app.config import settings
from app.routers import prediction

# Configure Logging: Output to console and file (logs/app.log)
log_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s"
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File handler (rotates at 5MB, keeps 5 backups)
file_log_path = os.path.join(settings.LOGS_DIR, "app.log")
file_handler = RotatingFileHandler(file_log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Initialize FastAPI App (disable Swagger and ReDoc docs)
app = FastAPI(
    docs_url=None,
    redoc_url=None
)

# Configure CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the results directory to serve explanation images statically
app.mount("/static/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")

# Register Prediction Router
app.include_router(prediction.router, tags=["Diagnosis & Explanations"])

@app.get("/", response_class=HTMLResponse)
async def root_index():
    """Returns the interactive clinical uploader dashboard."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health endpoint showing API operational status and model load state."""
    model_loaded = prediction.predictor_service.model is not None
    return {
        "status": "healthy",
        "service": "Brain Tumor Diagnosis API",
        "model_loaded": model_loaded,
        "model_type": settings.MODEL_TYPE,
        "device": "GPU (Accelerated)" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
    }
