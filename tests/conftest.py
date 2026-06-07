# Information: Pytest fixtures and session hooks.
# Importance: Programmatically creates a dummy Keras model with a 'top_conv' layer if no model exists so that test runs can pass immediately.

import os
import shutil
import tempfile
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi.testclient import TestClient

# Adjust path to find the app
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model():
    """
    Creates a small, fully functioning Keras model with a 'top_conv' layer 
    if no trained model exists. This allows tests to run without training.
    """
    model_path = settings.MODEL_PATH
    created_dummy = False

    if not os.path.exists(model_path):
        print(f"\n[TestSetup] No model detected. Creating dummy Keras model at: {model_path}")
        
        # Build mini CNN with 'top_conv' layer
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
        # Bounding box / Grad-CAM target layer
        x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="top_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        
        # Save to target file
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        created_dummy = True

    yield

    # Clean up the dummy model after the session finishes
    if created_dummy and os.path.exists(model_path):
        print(f"\n[TestTeardown] Cleaning up dummy Keras model from: {model_path}")
        try:
            os.remove(model_path)
        except Exception as e:
            print(f"Failed to delete dummy model: {str(e)}")

@pytest.fixture
def client():
    """Provides a TestClient for FastAPI endpoints."""
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def dummy_mri_image() -> Image.Image:
    """Generates a dummy 224x224 RGB image as a PIL Image."""
    # Create simple image with a white shape (suspected tumor) in center
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw a mock high-intensity circular lesion
    cv2_installed = True
    try:
        import cv2
        cv2.circle(arr, (112, 112), 30, (200, 200, 200), -1)
    except ImportError:
        # Draw manually if cv2 is not available (though it is in requirements)
        arr[82:142, 82:142, :] = 200
        
    return Image.fromarray(arr)

@pytest.fixture
def dummy_mri_bytes(dummy_mri_image) -> bytes:
    """Converts a PIL Image to raw PNG bytes."""
    import io
    img_byte_arr = io.BytesIO()
    dummy_mri_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()
