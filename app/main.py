# Information: Main application entry point for the FastAPI server.
# Importance: Registers routers, sets up CORS middleware, mounts results folder for static access, and configures rotating file loggers.

import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

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

# Initialize FastAPI App
app = FastAPI(
    title="Brain Tumor Detection & Explainable AI API",
    description="A production-ready Deep Learning API using TensorFlow/Keras to diagnose brain tumors from MRI scans, featuring Grad-CAM and Grad-CAM++ visualizations.",
    version="1.0.0"
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
    """Returns a premium landing page detailing API capabilities and usage."""
    # Premium Dark Mode HTML template
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Diagnosis API</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #0e1118 0%, #161b26 100%);
                color: #ffffff;
                font-family: 'Outfit', sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                align-items: center;
                justify-content: center;
            }
            .container {
                max-width: 800px;
                padding: 2.5rem;
                background: rgba(30, 30, 40, 0.65);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
                text-align: center;
            }
            h1 {
                font-size: 2.5rem;
                margin-top: 0;
                background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #4A90E2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p {
                color: #a0aabf;
                font-size: 1.1rem;
                line-height: 1.6;
            }
            .badge {
                display: inline-block;
                background: rgba(0, 255, 127, 0.15);
                color: #00FF7F;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9rem;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(0, 255, 127, 0.25);
            }
            .links {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 2rem;
            }
            a {
                background: #4A90E2;
                color: white;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            a:hover {
                background: #357ABD;
                transform: translateY(-2px);
            }
            a.docs {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
            a.docs:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            .footer {
                margin-top: 3rem;
                font-size: 0.85rem;
                color: #5d677a;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">API System Active</div>
            <h1>🧠 Brain Tumor Detection & Explainable AI</h1>
            <p>Welcome to the Brain Tumor Classification and Explainable Diagnosis backend. This API runs Transfer Learning inference using EfficientNet (B3/B0) and outputs Grad-CAM/Grad-CAM++ activation maps along with clinical symptoms and recommendations.</p>
            <div class="links">
                <a href="/docs">Interactive Swagger API Docs</a>
                <a href="/health" class="docs">Check Health Status</a>
            </div>
            <div class="footer">
                For educational and research purposes only. Not a medical diagnosis.
            </div>
        </div>
    </body>
    </html>
    """
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
