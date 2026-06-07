# Information: Inference service wrapping the loaded Keras model.
# Importance: Handles model instantiation, supports EfficientNet-B3/B0 switches, and runs predictions while handling missing model files gracefully.

import os
import logging
import numpy as np
import tensorflow as tf
from app.config import settings

logger = logging.getLogger(__name__)

def build_transfer_model(model_type: str, num_classes: int = 4, input_size: int = settings.IMAGE_SIZE) -> tf.keras.Model:
    """
    Builds the model architecture.
    
    Args:
        model_type: 'efficientnet_b0' or 'efficientnet_b3'
        num_classes: Number of classification targets.
        input_size: Image width/height.
        
    Returns:
        tf.keras.Model
    """
    input_shape = (input_size, input_size, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    if model_type.lower() == "efficientnet_b0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    elif model_type.lower() == "efficientnet_b3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'efficientnet_b0' or 'efficientnet_b3'.")
    
    # Freeze the base model layers by default
    base_model.trainable = False
    
    # Custom classification head matching the training notebook structure
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"BrainTumorClassifier_{model_type}")
    return model

class Predictor:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads model weights from the configured path."""
        path = settings.MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"Model file not found at: {path}. Predictions will fail until model is trained.")
            return

        try:
            # Try to load the full saved model first
            self.model = tf.keras.models.load_model(path, compile=False)
            logger.info(f"Model loaded successfully from: {path}")
        except Exception as e:
            logger.warning(f"Failed to load full model directly: {str(e)}. Re-building and loading weights...")
            try:
                # Fallback: Rebuild the architecture and load weights
                self.model = build_transfer_model(settings.MODEL_TYPE)
                self.model.load_weights(path)
                logger.info(f"Model weights loaded successfully after rebuilding architecture.")
            except Exception as ex:
                logger.error(f"Critical: Failed to load model weights: {str(ex)}")
                self.model = None

    def predict(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Runs inference on a preprocessed image batch."""
        if self.model is None:
            # Hot reload in case model was trained after startup
            self.load_model()
            if self.model is None:
                raise RuntimeError("Model is not loaded. Please train the model or provide the weights file.")
        
        # Call model directly for faster latency, clean logs, and to bypass Functional API input warnings
        predictions = self.model(preprocessed_image, training=False)
        return predictions[0].numpy()  # Return class probabilities (flat array)
