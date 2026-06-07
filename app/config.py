# Information: System-wide configuration module using Pydantic Settings.
# Importance: Defines limits (10MB payload size), allowed MRI formats, class mapping, and prepares logs/uploads directories on boot.

import os
from pydantic_settings import BaseSettings
from typing import Set, Dict

class Settings(BaseSettings):
    # Model configuration
    MODEL_TYPE: str = "efficientnet_b3"  # Choices: efficientnet_b0, efficientnet_b3
    MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5"))
    IMAGE_SIZE: int = 224  # Standard image size (224 for B0, can also be 224/300 for B3)
    
    # API / Security settings
    ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10 MB limit
    GEMINI_API_KEY: str = ""
    
    # Directories
    UPLOAD_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
    RESULTS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    LOGS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    
    # Class mappings (Alphabetical order: glioma, meningioma, notumor, pituitary)
    # 0 = glioma, 1 = meningioma, 2 = notumor, 3 = pituitary
    CLASS_MAPPING: Dict[int, str] = {
        0: "glioma",
        1: "meningioma",
        2: "notumor",
        3: "pituitary"
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate and ensure necessary folders exist
settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
