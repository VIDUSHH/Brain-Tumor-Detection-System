# Information: Main application entry point for the FastAPI server.
# Importance: Registers routers, sets up CORS middleware, mounts results folder for static access, and configures rotating file loggers.

# Configure environment variables for TensorFlow CPU memory optimization before imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
from logging.handlers import RotatingFileHandler
from app.config import settings

# Configure Logging: Output to console and file (logs/app.log) early
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

# Configure TensorFlow threading to use single core to save RAM on CPU instances
import tensorflow as tf
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError as e:
    logger.warning(f"Could not set TensorFlow threading parameters: {str(e)}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.routers import prediction

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
