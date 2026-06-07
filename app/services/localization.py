# Information: Tumor localization service using OpenCV.
# Importance: Performs binary mask thresholding on heatmaps and computes contours to draw red highlights and yellow bounding boxes around tumors.

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List

def generate_visualizations(
    original_image: Image.Image,
    heatmap_np: np.ndarray,
    threshold_ratio: float = 0.4,
    draw_boxes: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
    """
    Applies heatmaps, draws bounding boxes around activation hotspots,
    and returns localized images and coordinates.
    
    Args:
        original_image: PIL Image of the original scan.
        heatmap_np: 2D NumPy array (H, W) normalized between 0 and 1.
        threshold_ratio: Activation threshold (0.0 - 1.0) to segment the tumor.
        draw_boxes: Whether to draw bounding box and contour outlines.
        
    Returns:
        heatmap_colored: OpenCV BGR image of the colormapped heatmap.
        superimposed: OpenCV BGR image blending original + heatmap.
        localized: OpenCV BGR image with bounding box / contours.
        bbox_coords: Dictionary containing bounding box coordinates or None.
    """
    # Convert PIL original image to OpenCV BGR format
    orig_np = np.array(original_image)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    height, width, _ = orig_bgr.shape

    # 1. Detect empty heatmap (no tumor predicted or all zeros).
    # Return a solid blue heatmap and clean original scans to prevent false positives.
    if np.max(heatmap_np) < 1e-10:
        blank_heatmap = np.zeros_like(orig_bgr)
        # JET colormap maps zero to dark blue: B=128, G=0, R=0
        blank_heatmap[:, :] = [128, 0, 0]
        return blank_heatmap, orig_bgr, orig_bgr, None

    # 2. Resize heatmap using bicubic interpolation for smoother transitions
    heatmap_resized = cv2.resize(heatmap_np, (width, height), interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian smoothing to round out interpolation grid cells cleanly
    heatmap_smoothed = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)

    # Normalize again to ensure value range is strictly [0, 1] after interpolation
    max_val = np.max(heatmap_smoothed)
    if max_val > 1e-10:
        heatmap_smoothed = heatmap_smoothed / max_val

    # Apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap_smoothed)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap onto original image
    superimposed = cv2.addWeighted(orig_bgr, 0.65, heatmap_colored, 0.35, 0)

    localized = orig_bgr.copy()
    bbox_coords = None

    # 3. Only draw bounding box and contours if draw_boxes flag is enabled (tumor predicted)
    if draw_boxes:
        # Create thresholded binary mask using smoothed heatmap
        _, thresh = cv2.threshold(heatmap_uint8, int(255 * threshold_ratio), 255, cv2.THRESH_BINARY)

        # Morphological operations using a larger ellipse kernel to clean noise and small artifacts
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

        # Find contours in the cleaned binary mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_bbox = None
        best_contour = None

        # Filter contours by size, setting the minimum noise floor to 300 pixels
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300 and area > max_area:
                max_area = area
                x, y, w, h = cv2.boundingRect(contour)
                best_bbox = {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}
                best_contour = contour

        # Draw outlines and rectangle if a valid tumor region is segmented
        if best_bbox is not None:
            bbox_coords = best_bbox
            x, y, w, h = best_bbox["x_min"], best_bbox["y_min"], best_bbox["x_max"] - best_bbox["x_min"], best_bbox["y_max"] - best_bbox["y_min"]
            
            # Draw contour (Red) and bounding box (Yellow)
            cv2.drawContours(localized, [best_contour], -1, (0, 0, 255), 2)
            cv2.rectangle(localized, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Label overlay
            cv2.putText(
                localized,
                "Suspected Tumor Region",
                (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

    return heatmap_colored, superimposed, localized, bbox_coords
