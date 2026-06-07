import io
import pytest
from fastapi.testclient import TestClient

def test_root_index(client: TestClient):
    """Tests the home page route."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Brain Tumor Detection" in response.text

def test_health_check(client: TestClient):
    """Tests the health status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_type" in data

def test_predict_success(client: TestClient, dummy_mri_bytes: bytes):
    """Tests a successful tumor classification upload."""
    response = client.post(
        "/predict",
        files={"file": ("test_mri.png", io.BytesIO(dummy_mri_bytes), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify required keys in response JSON
    assert "prediction" in data
    assert "confidence" in data
    assert "all_scores" in data
    assert "explanation" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert "medical_disclaimer" in data
    assert "urls" in data
    assert "heatmap_url" in data
    assert "overlay_url" in data
    assert "localized_url" in data
    assert "bbox_coordinates" in data
    assert "latency_ms" in data

    # Verify probability formatting
    assert isinstance(data["all_scores"], dict)
    assert len(data["all_scores"]) == 4
    for score in data["all_scores"].values():
        assert isinstance(score, (int, float))

def test_predict_invalid_extension(client: TestClient):
    """Tests that uploading an invalid file extension returns a 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("malicious.txt", io.BytesIO(b"dummy text content"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_predict_file_too_large(client: TestClient):
    """Tests that files larger than 10MB are rejected with 413 Payload Too Large."""
    # Create a 10.5 MB block of mock bytes
    large_bytes = b"0" * (10 * 1024 * 1024 + 100 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("too_large.png", io.BytesIO(large_bytes), "image/png")}
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]
