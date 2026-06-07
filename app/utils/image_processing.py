import io
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from app.config import settings

def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Converts raw image bytes to PIL Image, ensuring RGB format."""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def preprocess_image(image: Image.Image, target_size: int = settings.IMAGE_SIZE) -> np.ndarray:
    """Preprocesses a PIL Image for EfficientNet models."""
    # Resize image
    image_resized = image.resize((target_size, target_size))
    # Convert to array
    image_array = np.array(image_resized, dtype=np.float32)
    # Expand dims to create batch dimension (1, H, W, C)
    image_batch = np.expand_dims(image_array, axis=0)
    # Apply EfficientNet preprocessing (which is generally a pass-through since rescale is inside)
    # but using keras preprocessing is best practice.
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(image_batch)
    return preprocessed

def save_result_image(image_cv: np.ndarray, filename: str) -> str:
    """Saves an OpenCV image (BGR) to the results directory and returns the absolute path."""
    dest_path = os.path.join(settings.RESULTS_DIR, filename)
    cv2.imwrite(dest_path, image_cv)
    return dest_path
