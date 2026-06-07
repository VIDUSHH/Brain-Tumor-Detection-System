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

    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap_np, (width, height))

    # Apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap onto original image
    superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Create thresholded binary mask
    _, thresh = cv2.threshold(heatmap_uint8, int(255 * threshold_ratio), 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    localized = orig_bgr.copy()
    bbox_coords = None
    max_area = 0
    best_bbox = None
    best_contour = None

    # Filter and find the largest contour area above a minimum noise floor (e.g., 100 pixels)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and area > max_area:
            max_area = area
            x, y, w, h = cv2.boundingRect(contour)
            best_bbox = {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}
            best_contour = contour

    # Draw markers if a tumor region is found
    if best_bbox is not None and draw_boxes:
        bbox_coords = best_bbox
        x, y, w, h = best_bbox["x_min"], best_bbox["y_min"], best_bbox["x_max"] - best_bbox["x_min"], best_bbox["y_max"] - best_bbox["y_min"]
        
        # Draw all contours that passed threshold (or just the largest)
        # Let's draw the largest contour for clarity and precision
        cv2.drawContours(localized, [best_contour], -1, (0, 0, 255), 2)  # Red contour
        cv2.rectangle(localized, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box
        
        # Put label text
        cv2.putText(
            localized,
            "Tumor Activation",
            (x, max(y - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return heatmap_colored, superimposed, localized, bbox_coords
