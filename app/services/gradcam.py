# Information: Explainable AI service implementing Grad-CAM and Grad-CAM++ algorithms in TensorFlow.
# Importance: Calculates gradients of prediction scores w.r.t the last convolutional layer output to output normalized visual heatmaps.

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional

class GradCAMExplainer:
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        """
        Explainer for generating Grad-CAM and Grad-CAM++ heatmaps.
        
        Args:
            model: The compiled Keras model.
            layer_name: Optional layer name to target. If None, auto-detects the last conv layer.
        """
        self.model = model
        self.layer_name = layer_name or self._auto_detect_last_conv()
        
        # Find the final Dense layer to reconstruct logits (avoids softmax gradient saturation)
        self.dense_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                self.dense_layer = layer
                break

        if self.dense_layer is None:
            raise ValueError("Could not find Dense layer in the model.")

        # Build a gradient model that outputs the target conv layer activations and Dense inputs
        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.dense_layer.input
            ]
        )

    def _auto_detect_last_conv(self) -> str:
        """Finds the name of the last Conv2D layer in the model."""
        # For standard Keras EfficientNet, the last conv layer is 'top_conv'
        try:
            if self.model.get_layer("top_conv"):
                return "top_conv"
        except ValueError:
            pass

        # Fallback: Traverse in reverse order to find the last Conv2D layer
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            # Handle nested models (like functional blocks)
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return sub_layer.name
        
        raise ValueError("Could not automatically locate a Conv2D layer in the model.")

    def generate_gradcam(self, img_tensor: np.ndarray, class_idx: int) -> np.ndarray:
        """Generates standard Grad-CAM heatmap using logits to avoid softmax saturation."""
        with tf.GradientTape() as tape:
            conv_outputs, last_layer_input = self.grad_model([img_tensor])
            # Reconstruct the logits manually
            logits = tf.matmul(last_layer_input, self.dense_layer.kernel) + self.dense_layer.bias
            loss = logits[:, class_idx]

        # Gradients of the active class score w.r.t the feature map activations
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients to get channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the 2D activation map by channel weights
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Apply ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize the heatmap
        max_val = tf.reduce_max(heatmap)
        if max_val > 1e-10:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def generate_gradcam_plusplus(self, img_tensor: np.ndarray, class_idx: int) -> np.ndarray:
        """Generates Grad-CAM++ heatmap using logits for sharper tumor localization."""
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, last_layer_input = self.grad_model([img_tensor])
                    # Reconstruct the logits manually
                    logits = tf.matmul(last_layer_input, self.dense_layer.kernel) + self.dense_layer.bias
                    loss = logits[:, class_idx]
                # First order gradients
                grads = tape3.gradient(loss, conv_outputs)
            # Second order gradients
            grads_2 = tape2.gradient(grads, conv_outputs)
        # Third order gradients
        grads_3 = tape1.gradient(grads_2, conv_outputs)

        # Numerical stability epsilon
        eps = 1e-10

        # Math formulations for Grad-CAM++
        # conv_outputs shape: (1, H, W, C)
        conv_outputs_val = conv_outputs[0]
        grads_val = grads[0]
        grads_2_val = grads_2[0]
        grads_3_val = grads_3[0]

        # Calculate alpha weights
        sum_activations = tf.reduce_sum(conv_outputs_val, axis=(0, 1))
        denominator = 2.0 * grads_2_val + sum_activations[tf.newaxis, tf.newaxis, :] * grads_3_val
        
        # Safe division
        alpha = grads_2_val / (denominator + eps)

        # Weight coefficient: sum of (alpha * ReLU(grads))
        weights = tf.reduce_sum(alpha * tf.maximum(grads_val, 0.0), axis=(0, 1))

        # Generate weighted feature activation map
        heatmap = tf.reduce_sum(conv_outputs_val * weights, axis=-1)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 1e-10:
            heatmap = heatmap / max_val

        return heatmap.numpy()
