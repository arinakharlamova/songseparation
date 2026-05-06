"""Integration tests for the FastAPI application."""
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_audio_file(tmp_path):
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    waveform = np.sin(2 * np.pi * 440 * t)
    file_path = tmp_path / "test_track.wav"
    sf.write(str(file_path), waveform, sr)
    return file_path


def test_root_endpoint(client):
    """Root endpoint serves web UI (HTML)."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_api_endpoint(client):
    """API info endpoint returns JSON."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Source Separation API"
    assert "model" in data
    assert "device" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_loaded" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["current"] == "htdemucs_ft"
    assert len(data["available"]) == 3
    names = [m["name"] for m in data["available"]]
    assert "htdemucs_ft" in names
    assert "htdemucs" in names
    assert "demucs" in names


def test_separate_invalid_extension(client, tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(b"not audio")
    with open(file_path, "rb") as f:
        response = client.post(
            "/separate",
            files={"file": ("test.txt", f, "text/plain")},
            data={"return_zip": "true"},
        )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_download_path_traversal(client):
    """Ensure path traversal is blocked."""
    response = client.get("/download/../../etc/passwd")
    assert response.status_code in (400, 403, 404)


def test_download_nonexistent_file(client):
    response = client.get("/download/nonexistent.wav")
    assert response.status_code == 404


def test_openapi_schema(client):
    """Verify OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/separate" in schema["paths"]
    assert "/models" in schema["paths"]
