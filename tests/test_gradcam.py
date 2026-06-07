import os
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
from app.services.predictor import Predictor
from app.services.gradcam import GradCAMExplainer
from app.services.localization import generate_visualizations

def test_gradcam_layer_autodetect():
    """Tests that the Grad-CAM explainer correctly targets the last Conv2D layer."""
    predictor = Predictor()
    explainer = GradCAMExplainer(predictor.model)
    
    # Check that it auto-detects the layer 'top_conv' from conftest dummy model
    assert explainer.layer_name == "top_conv"
    assert explainer.grad_model is not None

def test_gradcam_and_plusplus_generation(dummy_mri_image):
    """Tests that Grad-CAM and Grad-CAM++ produce valid normalized 2D heatmaps."""
    predictor = Predictor()
    explainer = GradCAMExplainer(predictor.model)
    
    # Preprocess
    img_resized = dummy_mri_image.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_arr, axis=0)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)
    
    # Generate heatmaps (for class index 0)
    heatmap_gc = explainer.generate_gradcam(preprocessed, class_idx=0)
    heatmap_gcpp = explainer.generate_gradcam_plusplus(preprocessed, class_idx=0)
    
    # Verify shape (conv output grid size is same shape as top_conv output spatial grid size)
    # The dummy model has 224x224 input, and does not pool/stride, so spatial size is 224x224
    assert len(heatmap_gc.shape) == 2
    assert len(heatmap_gcpp.shape) == 2
    
    # Check normalized properties (values between 0.0 and 1.0)
    assert heatmap_gc.min() >= 0.0
    assert heatmap_gc.max() <= 1.0
    assert heatmap_gcpp.min() >= 0.0
    assert heatmap_gcpp.max() <= 1.0

def test_localization_overlays_and_bbox(dummy_mri_image):
    """Tests that localization draws overlays and extracts correct bounding box formats."""
    # Generate mock heatmap with centered high activation spot
    heatmap_np = np.zeros((224, 224), dtype=np.float32)
    heatmap_np[90:130, 90:130] = 0.8  # high activation spot
    heatmap_np[100:120, 100:120] = 1.0  # max hotspot
    
    # Run localization with threshold of 40% (0.4)
    heatmap_colored, superimposed, localized, bbox = generate_visualizations(
        dummy_mri_image,
        heatmap_np,
        threshold_ratio=0.4,
        draw_boxes=True
    )
    
    # Dimensions check (matches dummy_mri_image: 224x224)
    assert heatmap_colored.shape == (224, 224, 3)
    assert superimposed.shape == (224, 224, 3)
    assert localized.shape == (224, 224, 3)
    
    # Bounding box assertions
    assert bbox is not None
    assert "x_min" in bbox
    assert "y_min" in bbox
    assert "x_max" in bbox
    assert "y_max" in bbox
    
    # Coordinates check
    assert bbox["x_min"] < bbox["x_max"]
    assert bbox["y_min"] < bbox["y_max"]
    assert bbox["x_min"] >= 0
    assert bbox["x_max"] <= 224
    assert bbox["y_min"] >= 0
    assert bbox["y_max"] <= 224
