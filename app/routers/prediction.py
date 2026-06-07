# Information: API prediction router implementing the POST /predict endpoint.
# Importance: Coordinates file upload security checks, pre-processing, Grad-CAM overlays, contour bounding box extractions, and explanation mappings.

import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2

from app.config import settings
from app.services.predictor import Predictor
from app.services.gradcam import GradCAMExplainer
from app.services.localization import generate_visualizations
from app.services.explanation_engine import ExplanationEngine
from app.utils.image_processing import load_image_from_bytes, preprocess_image, save_result_image
from app.utils.metrics import LatencyTracker, format_probabilities

logger = logging.getLogger(__name__)
router = APIRouter()

# Global predictor and explanation engine instances
# Predictor is initialized once, loading model weights if present
predictor_service = Predictor()
explanation_engine = ExplanationEngine()

def get_predictor():
    return predictor_service

@router.post("/predict")
async def predict_mri(
    request: Request,
    file: UploadFile = File(...),
    predictor: Predictor = Depends(get_predictor)
):
    """
    Predicts brain tumor type from a patient's MRI scan.
    Generates Grad-CAM and Grad-CAM++ heatmaps, bounding box coordinates, and clinical explanations.
    """
    logger.info(f"Received prediction request for file: {file.filename}")

    # 1. Security check: Validate file extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected: Invalid extension '{ext}'")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # 2. Security check: Validate file size (10MB limit)
    # Read bytes to verify length
    file_bytes = await file.read()
    file_size = len(file_bytes)
    if file_size > settings.MAX_CONTENT_LENGTH:
        logger.warning(f"File upload rejected: Size {file_size} bytes exceeds 10MB limit")
        raise HTTPException(
            status_code=413,
            detail="File size exceeds maximum allowed limit of 10MB."
        )

    # 3. Handle malformed images
    try:
        pil_image = load_image_from_bytes(file_bytes)
    except Exception as e:
        logger.error(f"Malformed image upload: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image or is corrupted."
        )

    # Track latency for inference & processing
    with LatencyTracker() as tracker:
        try:
            # 4. Preprocess for EfficientNet
            preprocessed = preprocess_image(pil_image)

            # 5. Run Prediction
            # If the model is not loaded yet (not trained), this raises a RuntimeError
            probabilities = predictor.predict(preprocessed)
            
            # Map probabilities
            pred_class_idx = int(np.argmax(probabilities))
            pred_class_name = settings.CLASS_MAPPING[pred_class_idx]
            confidence_score = float(probabilities[pred_class_idx])

            # Convert numpy array probability breakdown to dictionary
            all_scores = format_probabilities(probabilities, settings.CLASS_MAPPING)

            # 6. Generate Explainable AI (Grad-CAM and Grad-CAM++)
            # Grad-CAM auto-detects target layer inside the class
            explainer = GradCAMExplainer(predictor.model)
            
            heatmap_gc = explainer.generate_gradcam(preprocessed, pred_class_idx)
            heatmap_gcpp = explainer.generate_gradcam_plusplus(preprocessed, pred_class_idx)

            # Determine whether tumor is detected (class is not 'notumor')
            # 0=glioma, 1=meningioma, 2=notumor, 3=pituitary
            is_tumor = (pred_class_name != "notumor")

            # 7. Localize the tumor region and save files
            # For localization, we use the sharper Grad-CAM++ heatmap
            heatmap_col_gc, overlay_gc, _, _ = generate_visualizations(
                pil_image, heatmap_gc, threshold_ratio=0.4, draw_boxes=False
            )
            
            heatmap_col_gcpp, overlay_gcpp, localized_gcpp, bbox_coords = generate_visualizations(
                pil_image, heatmap_gcpp, threshold_ratio=0.4, draw_boxes=is_tumor
            )

            # Save generated images with unique IDs
            req_id = str(uuid.uuid4())[:8]
            
            # Save original file temporarily for trace
            orig_filename = f"original_{req_id}.jpg"
            orig_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            save_result_image(orig_cv, orig_filename)

            # Save explainers
            save_result_image(heatmap_col_gc, f"heatmap_gradcam_{req_id}.jpg")
            save_result_image(overlay_gc, f"overlay_gradcam_{req_id}.jpg")
            save_result_image(heatmap_col_gcpp, f"heatmap_gradcampp_{req_id}.jpg")
            save_result_image(overlay_gcpp, f"overlay_gradcampp_{req_id}.jpg")
            save_result_image(localized_gcpp, f"localization_{req_id}.jpg")

            # 8. Generate Medical Explanation
            explanation_data = explanation_engine.generate_explanation(
                pred_class_name, confidence_score, all_scores
            )

            # Check if Gemini API key is present in headers or settings to run generative summarization
            # (Allows dynamic API key pass-through from client headers)
            client_gemini_key = request.headers.get("X-Gemini-API-Key") or settings.GEMINI_API_KEY
            if client_gemini_key:
                gemini_engine = ExplanationEngine(api_key=client_gemini_key)
                explanation_data["generative_explanation"] = gemini_engine.generate_ai_explanation(
                    pred_class_name, confidence_score, all_scores
                )

            # Build URLs for the saved static images
            # FastAPI serves the results/ directory under /static/results
            base_url = str(request.base_url).rstrip("/")
            
            response_payload = {
                "prediction": pred_class_name,
                "confidence": round(confidence_score * 100, 2),
                "all_scores": {k: round(v * 100, 2) for k, v in all_scores.items()},
                "explanation": explanation_data["reasoning"],
                "risk_level": explanation_data["risk_level"],
                "recommendation": explanation_data["recommendation"],
                "medical_disclaimer": explanation_data["medical_disclaimer"],
                "tumor_characteristics": explanation_data["characteristics"],
                "potential_symptoms": explanation_data["potential_symptoms"],
                "bbox_coordinates": bbox_coords,
                "latency_ms": round(tracker.duration_ms, 2),
                "urls": {
                    "original_url": f"{base_url}/static/results/{orig_filename}",
                    "heatmap_gradcam_url": f"{base_url}/static/results/heatmap_gradcam_{req_id}.jpg",
                    "overlay_gradcam_url": f"{base_url}/static/results/overlay_gradcam_{req_id}.jpg",
                    "heatmap_gradcampp_url": f"{base_url}/static/results/heatmap_gradcampp_{req_id}.jpg",
                    "overlay_gradcampp_url": f"{base_url}/static/results/overlay_gradcampp_{req_id}.jpg",
                    "localized_url": f"{base_url}/static/results/localization_{req_id}.jpg"
                }
            }

            # Compatibility: map standard fields directly at the root as requested by user
            response_payload["heatmap_url"] = response_payload["urls"]["heatmap_gradcam_url"]
            response_payload["overlay_url"] = response_payload["urls"]["overlay_gradcam_url"]
            response_payload["localized_url"] = response_payload["urls"]["localized_url"]

            logger.info(f"Successful diagnosis: {pred_class_name} ({response_payload['confidence']}% confidence) in {response_payload['latency_ms']}ms")
            return response_payload

        except RuntimeError as re:
            logger.error(f"Prediction error: {str(re)}")
            raise HTTPException(
                status_code=503,
                detail="The diagnostic model is currently not loaded. Please ensure the model is trained."
            )
        except Exception as e:
            logger.error(f"Inference pipeline failure: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred during the MRI diagnostic analysis."
            )
