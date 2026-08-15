import numpy as np
import tensorflow as tf
from typing import Optional

class GradCAMExplainer:
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        """
        Explainer for generating Grad-CAM heatmaps.

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

    def _normalize_heatmap(self, heatmap: tf.Tensor) -> tf.Tensor:
        """Normalizes a raw heatmap to the [0, 1] range using min-max scaling."""
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 1e-10:
            heatmap = heatmap / max_val
        return heatmap

    def generate_gradcam(self, img_tensor: np.ndarray, class_idx: int, power: float = 1.0) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap using logits to avoid softmax saturation.

        Args:
            img_tensor: Preprocessed image batch (1, H, W, 3).
            class_idx: Index of the target class.
            power: Exponent used to sharpen the heatmap. Values > 1 focus on the
                   strongest activation clusters, which improves tumor localization.

        Returns:
            heatmap: 2D NumPy array (H, W) normalized to [0, 1].
        """
        with tf.GradientTape() as tape:
            conv_outputs, last_layer_input = self.grad_model([img_tensor])
            # Reconstruct the logits manually
            logits = tf.matmul(last_layer_input, self.dense_layer.kernel) + self.dense_layer.bias
            loss = logits[:, class_idx]

        # Gradients of the active class score w.r.t the feature map activations
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError(f"Gradients could not be computed for layer: {self.layer_name}")

        # Keep only positive gradients (negative gradients point away from the class)
        grads = tf.maximum(grads, 0.0)

        # Global average pooling of gradients to get channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the 2D activation map by channel weights
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Apply ReLU and normalize to [0, 1]
        heatmap = self._normalize_heatmap(heatmap)

        # Optional sharpening to emphasize the hottest activation cluster
        if power > 1.0:
            heatmap = self._normalize_heatmap(tf.pow(heatmap, power))

        return heatmap.numpy()
