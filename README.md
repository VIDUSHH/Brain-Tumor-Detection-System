# Brain Tumor Detection & Explainable AI System

An advanced, production-grade Deep Learning and Explainable AI (XAI) system for detecting, classifying, and localizing brain tumors from MRI scans using TensorFlow/Keras and FastAPI.

This system classifies scans into four distinct clinical categories (**Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**) and provides transparent visual explanations of its decisions using **Grad-CAM** heatmaps alongside contour outlines and bounding boxes. If configured, it also leverages the **Google Gemini API** to translate clinical findings into warm, patient-friendly summaries.

---

## Features

- **Multi-Class Tumor Classification**: Classifies brain MRI scans into 4 categories: Glioma, Meningioma, Pituitary, or No Tumor.
- **Switchable Deep Learning Architectures**: Configurable transfer learning backbones supporting `EfficientNet-B3` (default) and `EfficientNet-B0`.
- **Explainable AI (XAI) Visualizations**: Generates pixel-level heatmap overlays highlighting exactly where the neural network focused using **Grad-CAM**, with an optional sharpening power to emphasize the strongest activation clusters.
- **Tumor Localization**: Auto-detects the hottest activation cluster from the Grad-CAM heatmap using adaptive percentile thresholding and energy-weighted contour selection, then draws a **red contour outline** around the suspected lesion and a **yellow bounding box** with coordinates.
- **Generative Patient Explanations**: Fully integrates Google Gemini (via `google-generativeai`) to convert complex clinical probabilities, pathology traits, and risks into empathetic patient-friendly summaries.
- **Production-Grade API Backend**: FastAPI with strict payload size validation (10MB maximum), file-extension allow-list, image integrity verification, CORS enablement, and rotating file logs (`logs/app.log`, 5MB × 5 backups).
- **Graceful Error Handling**: Returns clear **503** responses when the model isn't trained/loaded, **413** for oversized files, and **400** for invalid/corrupt uploads.
- **One-Click Cloud Deployment**: Pre-configured Docker builds (`Dockerfile` + `docker-compose.yml`) with deployment blueprints for **Render** (`render.yaml`) and a **Hugging Face Spaces** build (`deploy/huggingface/`).

---

## Project Structure

```
Brain-Tumor_Project/
├── app/
│   ├── main.py                  # FastAPI app entry, CORS, static results mount, rotating loggers
│   ├── config.py                # Pydantic settings (model, paths, limits, class mapping)
│   ├── routers/
│   │   └── prediction.py        # POST /predict endpoint & pipeline orchestration
│   ├── services/
│   │   ├── predictor.py         # Model loading + inference (hot reload fallback)
│   │   ├── gradcam.py           # Grad-CAM explainer (logits-based gradients, sharpening)
│   │   ├── localization.py      # Heatmap overlay, contours, bounding boxes
│   │   └── explanation_engine.py# Clinical knowledge-base + optional Gemini summaries
│   ├── utils/
│   │   ├── image_processing.py  # Load / preprocess / save images
│   │   └── metrics.py           # Latency tracker, probability formatting
│   └── templates/index.html     # Interactive uploader dashboard
├── training/
│   ├── data_loader.py           # tf.data loaders + class weights
│   └── train.ipynb              # Two-stage transfer-learning training notebook
├── tests/                       # pytest suite (API, Grad-CAM, model)
├── models/best_model.h5         # Trained weights (created by training)
├── uploads/                     # Uploaded files
├── results/                     # Generated images (served at /static/results)
├── logs/app.log                 # Rotating application log
├── Dockerfile                   # Python 3.11-slim, Gunicorn + Uvicorn (bundles the model)
├── docker-compose.yml           # One-command local container run (port 8000)
├── .dockerignore                # Keeps the Docker build context lean
├── deploy/huggingface/          # HF Spaces Docker build + automated deploy script
├── render.yaml                  # Render blueprint (Docker runtime)
└── requirements.txt
```

---

## Dataset Structure

Place your brain tumor MRI dataset in the root of the project folder using the following structure:

```
Brain Tumor data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

- **Classes** (alphabetical order matches `image_dataset_from_directory`):
  - `0 = glioma` (Intra-axial brain tumor)
  - `1 = meningioma` (Extra-axial meningeal tumor)
  - `2 = notumor` (Normal brain MRI scan)
  - `3 = pituitary` (Sellar region pituitary adenoma)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/VIDUSHH/Brain-Tumor-Detection-System.git
cd Brain-Tumor-Detection-System
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
```
Activate it:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on dependency versions:** The project is built and tested against the TensorFlow 2.15 / Keras 2.15 stack. If you encounter import errors after installing (e.g., `cannot import name 'pywrap_tensorflow'` or `RecursionError`), pin the compatible set:
> ```bash
> pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4 protobuf==4.25.9
> ```

---

## Training the Model

Model training is executed using the interactive Jupyter Notebook [training/train.ipynb](training/train.ipynb). The training pipeline implements random horizontal/vertical flips, rotations, translations, zoom and contrast augmentation, class weight balancing, early stopping, learning-rate reduction, and automatic model checkpointing. It also enables mixed-precision float16 training automatically when a GPU is detected.

To train the model:
1. Open the [training/train.ipynb](training/train.ipynb) notebook.
2. Select your Python virtual environment kernel.
3. Configure the `MODEL_TYPE` variable in the configuration block (choose `"efficientnet_b3"` or `"efficientnet_b0"`).
4. Run all cells to execute **Stage 1** (head training at LR 1e-3, base frozen) and **Stage 2** (base model fine-tuning at LR 1e-5, BatchNorm layers kept frozen for stability).
5. The best weights are saved automatically to `models/best_model.h5` via a `ModelCheckpoint` callback.

### Training Configuration (defaults)
| Parameter        | Value                              |
|------------------|------------------------------------|
| Backbone         | `efficientnet_b3` (or `efficientnet_b0`) |
| Input size       | 224 × 224 × 3                      |
| Batch size       | 32                                 |
| Head epochs      | 10 (LR 1e-3)                       |
| Fine-tune epochs | 15 (LR 1e-5)                       |
| Output head      | Dense(256, ReLU) → Dropout(0.3) → Dense(4, softmax, float32) |

---

## Running the Application

Start the FastAPI application locally:
```bash
uvicorn app.main:app --reload
```

- **Local Address**: `http://127.0.0.1:8000`
- **Uploader Dashboard**: `http://127.0.0.1:8000/` (interactive HTML demo)
- **Health Check**: `http://127.0.0.1:8000/health` (service status, model load state, device)

> Interactive Swagger/ReDoc docs are disabled in production (`docs_url=None`) to keep the public surface minimal.

> **Important:** The prediction endpoint requires a trained model at `models/best_model.h5`. If the file is missing, the API still boots but `/predict` returns **503** with a clear message. Train the model first (see above).

### Running with Docker

The project ships a `Dockerfile`, `.dockerignore`, and `docker-compose.yml`. The image **bundles the trained model** (`models/best_model.h5`), so the container works out of the box.

Build and run with Docker Compose (single command):
```bash
docker compose up --build
```

Or manually:
```bash
docker build -t brain-tumor-detection:latest .
docker run -d --name btd-app -p 8000:10000 \
  -v "$(pwd)/results:/workspace/results" \
  -v "$(pwd)/logs:/workspace/logs" \
  brain-tumor-detection:latest
```

- App: `http://127.0.0.1:8000/` | Health: `http://127.0.0.1:8000/health`
- The `results/` and `logs/` folders are bind-mounted so generated images and logs persist on the host.
- The image listens on port **10000** (host port mapped to **8000**). Requires Docker (Docker Desktop on Windows/macOS). First build takes a few minutes while TensorFlow installs.

> **Note:** only the `results/` and `logs/` folders are mounted — uploaded images and generated heatmaps inside the container are ephemeral unless you also mount `uploads/`.

---

## Example API Response

### `POST /predict`
Submit an MRI image to the `/predict` endpoint (as `multipart/form-data`, field name `file`).

#### Request Header (Optional for AI Patient Summaries)
- `X-Gemini-API-Key`: `your_gemini_api_key_here` (or define `GEMINI_API_KEY` in the environment)

#### Response JSON
```json
{
  "prediction": "glioma",
  "confidence": 98.45,
  "all_scores": {
    "glioma": 98.45,
    "meningioma": 0.82,
    "notumor": 0.12,
    "pituitary": 0.61
  },
  "explanation": "The model detected abnormal high-intensity tissue patterns in the brain parenchyma. Grad-CAM highlighted concentrated activation around the suspected lesion area. The detected morphology resembles common Glioma characteristics including irregular margins and infiltrative growth patterns.",
  "risk_level": "High to Critical",
  "recommendation": "Urgent consultation with a neurologist or neurosurgeon for MRI review. Schedule a contrast-enhanced brain MRI (with spectroscopy or perfusion if recommended by the specialist). Consult a neuro-oncologist to review options for surgical biopsy/resection, radiation, and chemotherapy.",
  "medical_disclaimer": "This prediction is generated by an AI model and should not be considered a medical diagnosis. Consult a neurologist or neurosurgeon for professional MRI review.",
  "tumor_characteristics": [
    "Originates from glial cells (astrocytes, oligodendrocytes, ependymal cells)",
    "Grows intra-axially (inside the brain tissue) with infiltrative, irregular margins",
    "Can range from low-grade (slow-growing) to high-grade (highly aggressive, e.g., Glioblastoma)"
  ],
  "potential_symptoms": [
    "Headaches (especially worse in the morning)",
    "Seizures",
    "Vision problems or double vision",
    "Memory loss, confusion, or personality changes"
  ],
  "bbox_coordinates": {
    "x_min": 78,
    "y_min": 52,
    "x_max": 154,
    "y_max": 128
  },
  "latency_ms": 345.12,
  "urls": {
    "original_url": "http://127.0.0.1:8000/static/results/original_827f8a.jpg",
    "heatmap_gradcam_url": "http://127.0.0.1:8000/static/results/heatmap_gradcam_827f8a.jpg",
    "overlay_gradcam_url": "http://127.0.0.1:8000/static/results/overlay_gradcam_827f8a.jpg",
    "localized_url": "http://127.0.0.1:8000/static/results/localization_827f8a.jpg"
  },
  "heatmap_url": "http://127.0.0.1:8000/static/results/heatmap_gradcam_827f8a.jpg",
  "overlay_url": "http://127.0.0.1:8000/static/results/overlay_gradcam_827f8a.jpg",
  "localized_url": "http://127.0.0.1:8000/static/results/localization_827f8a.jpg",
  "generative_explanation": "*Empathy-driven Gemini breakdown of the findings...*"
}
```

> **Notes:**
> - `bbox_coordinates` is `null` when no tumor is predicted (a `notumor` scan).
> - `generative_explanation` is only present when a Gemini API key is configured.
> - All URLs are served from the `results/` directory mounted at `/static/results`.

---

## Grad-CAM Visualizations & Localization

The system outputs multiple files in the `results/` folder for every prediction:

1. **Heatmap** (`heatmap_gradcam_<id>.jpg`): A JET-colormapped image visualizing pixel-level importance. Red areas indicate regions that strongly influenced the model's prediction; blue regions indicate negligible influence.
2. **Overlay** (`overlay_gradcam_<id>.jpg`): Blends the JET heatmap (35% weight) with the original MRI scan (65% weight). This lets radiologists map model activations back to precise brain anatomical structures (e.g., ventricles, frontal lobes, pituitary fossa).
3. **Localization** (`localization_<id>.jpg`): Segments the tumor by thresholding the Grad-CAM heatmap with an **adaptive percentile threshold** (keeps at least the top 25% of activated pixels, never below the 45% floor), cleans it with morphological open/close operations, and scores each contour by mean activation energy. The contour containing the **global activation peak** wins, which prevents mis-localization to large but diffuse regions. It then draws a **red contour outline** plus a **yellow bounding box** labeled "Suspected Tumor Region".

### Implementation details
- The explainer **auto-detects the last Conv2D layer** (`top_conv` for EfficientNet, with a reverse-scan fallback) so it works across switchable backbones.
- **Logits are reconstructed** as `matmul(dense_input, dense_kernel) + dense_bias` and differentiated *before* softmax, avoiding gradient saturation from the softmax output.
- Only **positive gradients** are kept before channel-weight pooling, so channels that point *away* from the predicted class do not dilute the heatmap.
- An optional **sharpening power** (`power > 1`) is applied after normalization to concentrate the map on the strongest activation cluster — used at `power=2.0` for localization while preserving the argmax location.
- When `notumor` is predicted, **zero heatmaps** are returned to prevent misleading false-positive activations; the localization step then returns a clean, unmodified scan.

---

## Running Tests

The project ships a pytest suite covering the API, Grad-CAM, and model layers:

```bash
pytest tests/ -v
```

The tests use a session-scoped dummy-model fixture (built in `tests/conftest.py`) so the full suite runs **even without a trained model**:
- `tests/test_api.py` — root page, `/health`, successful `/predict`, invalid-extension (400), oversized-file (413).
- `tests/test_gradcam.py` — layer auto-detection, Grad-CAM heatmap shape/range and sharpening, localization overlays, bounding boxes, and peak-preference over diffuse regions.
- `tests/test_model.py` — transfer model architecture and inference output.

---

## Render Deployment

This project contains a `render.yaml` specification that allows you to easily spin up a Dockerized FastAPI service on Render.

### Steps to Deploy
1. Push this repository to your GitHub account.
2. Log in to your **[Render Dashboard](https://dashboard.render.com/)**.
3. Click **New +** and select **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically parse the `render.yaml` file and configure a Web Service:
   - **Environment**: Docker
   - **Plan**: Starter (sufficient RAM for TensorFlow memory overhead)
   - **Runtime**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:10000`
6. Add your environment variables in the Render console:
   - `GEMINI_API_KEY`: *(Optional)* Your Google Gemini API key to enable generative summaries.
   - `MODEL_TYPE`: `efficientnet_b3` *(Default)* or `efficientnet_b0`.
   - `TF_CPP_MIN_LOG_LEVEL`: `2` *(quiet TensorFlow logging)*
7. Click **Apply** to build and launch your container.

> **Note:** Since the trained model is not committed to the repository, upload `models/best_model.h5` to your deployed instance (or bind storage) after the first build. Alternatively, build from a checkout that contains the weights — the `Dockerfile` bundles `models/best_model.h5` into the image automatically.

> **Note:** Render's **free** tier sleeps after ~15 minutes of inactivity and takes 1–2 minutes to cold-start. A paid instance type keeps the service always-on. If you need always-on hosting, consider Docker Compose on your own machine or a paid plan.

---

## Environment Variables

| Variable               | Default            | Description                                                          |
|------------------------|--------------------|----------------------------------------------------------------------|
| `MODEL_TYPE`           | `efficientnet_b3`  | Backbone architecture (`efficientnet_b3` or `efficientnet_b0`)       |
| `GEMINI_API_KEY`       | *(empty)*          | Google Gemini key for patient-friendly AI summaries                  |
| `IMAGE_SIZE`           | `224`              | Model input resolution                                               |
| `TF_CPP_MIN_LOG_LEVEL` | *(unset)*          | TensorFlow log verbosity (set `2` in production)                     |

Settings are read from environment variables or an optional `.env` file via Pydantic `BaseSettings`.

---

## Disclaimer

> **IMPORTANT CLINICAL DISCLAIMER**: This application is for **educational, demonstration, and research purposes only**. The predictions, heatmap localizations, and explanations generated by this model are not medical diagnoses, should not be treated as professional medical advice, and should never be used as a substitute for clinical evaluations. All medical images and findings must be reviewed and verified by a licensed Radiologist, Neurologist, or Neurosurgical specialist.
