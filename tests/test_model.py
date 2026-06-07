import os
import pytest
import numpy as np
import tensorflow as tf
from app.services.predictor import build_transfer_model, Predictor
from app.config import settings

def test_build_transfer_model():
    """Tests that the transfer learning model builder produces the correct structure."""
    model = build_transfer_model("efficientnet_b0", num_classes=4, input_size=224)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 4)
    assert model.input_shape == (None, 224, 224, 3)
    
    # Verify base model layers are frozen
    # We find the base model layer (named efficientnetb0) and check trainable status
    base_layer = next((l for l in model.layers if "efficientnet" in l.name), None)
    assert base_layer is not None
    assert base_layer.trainable is False

def test_predictor_runs_inference(dummy_mri_image):
    """Tests the Predictor service loading and running prediction on preprocessed image."""
    predictor = Predictor()
    
    # Ensure model loaded successfully from conftest setup
    assert predictor.model is not None
    
    # Preprocess dummy image
    img_resized = dummy_mri_image.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_arr, axis=0)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)
    
    # Run prediction
    probs = predictor.predict(preprocessed)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    # Check that predictions sum to ~1 (softmax distribution)
    assert np.allclose(np.sum(probs), 1.0, atol=1e-5)
    # Check that probabilities are between 0 and 1
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
