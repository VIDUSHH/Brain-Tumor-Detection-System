import google.generativeai as genai
import os
from typing import Dict, Any, Optional

CLINICAL_KNOWLEDGE = {
    "glioma": {
        "full_name": "Glioma (Intra-axial Brain Tumor)",
        "definition": "A glioma is a type of tumor that starts in the glial cells (cells that surround and support nerve cells) of the brain or spine.",
        "characteristics": [
            "Originates from glial cells (astrocytes, oligodendrocytes, ependymal cells)",
            "Grows intra-axially (inside the brain tissue) with infiltrative, irregular margins",
            "Can range from low-grade (slow-growing) to high-grade (highly aggressive, e.g., Glioblastoma)"
        ],
        "symptoms": [
            "Headaches (especially worse in the morning)",
            "Seizures",
            "Vision problems or double vision",
            "Memory loss, confusion, or personality changes"
        ],
        "risk_level": "High to Critical",
        "recommendations": [
            "Urgent consultation with a neurologist or neurosurgeon for MRI review.",
            "Schedule a contrast-enhanced brain MRI (with spectroscopy or perfusion if recommended by the specialist).",
            "Consult a neuro-oncologist to review options for surgical biopsy/resection, radiation, and chemotherapy."
        ],
        "reasoning": "The model detected abnormal high-intensity tissue patterns in the brain parenchyma. Grad-CAM/Grad-CAM++ highlighted concentrated activation around the suspected lesion area. The detected morphology resembles common Glioma characteristics including irregular margins and infiltrative growth patterns."
    },
    "meningioma": {
        "full_name": "Meningioma (Extra-axial Brain Tumor)",
        "definition": "A meningioma is a tumor that arises from the meninges—the outer membranes that cover and protect the brain and spinal cord.",
        "characteristics": [
            "Originates from the meningeal layers surrounding the brain",
            "Grows extra-axially (outside the brain tissue itself), compressing adjacent structures",
            "Most meningiomas (80-90%) are slow-growing and benign (WHO Grade I)"
        ],
        "symptoms": [
            "Gradually worsening headaches",
            "Localized seizures",
            "Weakness or numbness in arms or legs",
            "Hearing loss or ringing in the ears"
        ],
        "risk_level": "Moderate to High",
        "recommendations": [
            "Schedule a consultation with a neurosurgeon to evaluate the tumor's size and mass effect.",
            "If small and asymptomatic, a 'watch-and-wait' strategy with serial surveillance MRIs (every 6-12 months) may be appropriate.",
            "Surgical resection is the primary treatment for symptomatic or growing meningiomas."
        ],
        "reasoning": "The model identified a well-delineated, extra-axial mass margin. Grad-CAM activations are focused along the outer protective layers of the brain, suggesting a dural-based lesion pushing adjacent brain tissue aside."
    },
    "pituitary": {
        "full_name": "Pituitary Tumor (Sellar Region Adenoma)",
        "definition": "A pituitary tumor is an abnormal growth that develops in the pituitary gland, a pea-sized organ at the base of the brain.",
        "characteristics": [
            "Originates in the pituitary fossa (sella turcica) at the skull base",
            "Almost all pituitary tumors are benign adenomas (Grade I)",
            "Can cause significant endocrine (hormonal) disorders and compress the optic chiasm"
        ],
        "symptoms": [
            "Vision loss, particularly peripheral vision loss (bitemporal hemianopsia)",
            "Hormonal imbalances (infertility, weight changes, unexplained fatigue)",
            "Headaches located behind the eyes"
        ],
        "risk_level": "Moderate to High",
        "recommendations": [
            "Consultation with both an endocrinologist (to evaluate hormone levels) and a neurosurgeon.",
            "Complete a blood hormone panel (checking Prolactin, Cortisol, ACTH, TSH, Growth Hormone).",
            "Perform a formal visual field test to assess optical chiasm compression."
        ],
        "reasoning": "The model detected a midline, sellar/suprasellar anatomical expansion at the base of the brain. The Grad-CAM heatmap focuses precisely on the skull base region, typical of pituitary adenoma location."
    },
    "notumor": {
        "full_name": "No Tumor Detected (Normal Brain Anatomy)",
        "definition": "The MRI scan shows normal brain structure with no signs of neoplastic growths, abnormal contrast enhancement, or mass effect.",
        "characteristics": [
            "Symmetrical ventricles and normal tissue density",
            "Intact midline structures and clear sulci/gyri",
            "Absence of abnormal enhancements, swelling, or lesions"
        ],
        "symptoms": [
            "If experiencing headaches, dizziness, or neurological signs, non-tumor causes should be evaluated."
        ],
        "risk_level": "Normal",
        "recommendations": [
            "Routine health follow-ups.",
            "If symptoms persist, consult a neurologist to investigate non-neoplastic causes (e.g., migraines, tension, vascular issues)."
        ],
        "reasoning": "The model shows diffuse, low-intensity, non-localized activations spread across standard anatomical boundaries. No focal hyper-intensities or mass displacements were identified."
    }
}

class ExplanationEngine:
    """Generates structured medical explanations and AI-driven summaries."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def generate_explanation(self, prediction: str, confidence: float, all_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generates a structured medical-style explanation using clinical knowledge."""
        prediction = prediction.lower()
        if prediction not in CLINICAL_KNOWLEDGE:
            raise ValueError(f"Unknown prediction class: {prediction}")

        info = CLINICAL_KNOWLEDGE[prediction]
        
        return {
            "prediction": prediction.capitalize(),
            "confidence": f"{confidence * 100:.1f}%",
            "all_scores": {k: float(v) for k, v in all_scores.items()},
            "reasoning": info["reasoning"],
            "characteristics": info["characteristics"],
            "potential_symptoms": info["symptoms"],
            "risk_level": info["risk_level"],
            "recommendation": " ".join(info["recommendations"]),
            "medical_disclaimer": "This prediction is generated by an AI model and should not be considered a medical diagnosis. Consult a neurologist or neurosurgeon for professional MRI review."
        }

    def generate_ai_explanation(self, prediction: str, confidence: float, all_scores: Dict[str, float]) -> str:
        """Generates an empathetic, patient-friendly explanation using Gemini API if key is present."""
        if not self.api_key:
            return "AI Explanation: Gemini API Key not configured. Provide GEMINI_API_KEY in environment to enable."

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prediction = prediction.lower()
            info = CLINICAL_KNOWLEDGE[prediction]
            scores_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in all_scores.items()])
            
            prompt = f"""
            You are an empathetic, professional medical assistant. A machine learning model classified a patient's brain MRI scan.
            
            Here are the details:
            - Predicted Diagnosis: {info['full_name']}
            - Model Confidence: {confidence * 100:.1f}%
            - Class Probabilities: {scores_str}
            - Severity Level: {info['risk_level']}
            - Reasoning: {info['reasoning']}
            - Pathology description: {info['definition']}
            - Suggested actions: {", ".join(info['recommendations'])}
            
            Please write a patient-friendly, empathetic breakdown of this result.
            Explain:
            1. What this means in simple, non-medical language.
            2. Reassure the patient (especially if it is benign like a meningioma or pituitary tumor, or explain the critical urgency gently if it's a glioma).
            3. Clarify why the model predicted this specific type over others (differential reasoning in lay terms).
            4. Detail what their immediate next steps are, including seeing a doctor, and why it is important.
            5. Keep the tone warm, clear, and supportive. Use bullet points for readability. Include a clear disclaimer that they should consult their doctor.
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating AI explanation: {str(e)}"
