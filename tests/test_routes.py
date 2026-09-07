import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module


class FakeDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = device

    def detect(self, image_bgr, conf=0.35):
        return [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    with TestClient(main_module.app) as test_client:
        yield test_client


def make_test_image_bytes():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_status_endpoint_reports_model_loaded(client):
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["cameras_calibrated"] == []


def test_calibrate_endpoint_returns_homography(client):
    payload = {
        "camera_id": "test-cam",
        "points": [
            {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
            {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
            {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
            {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
        ],
    }
    response = client.post("/calibrate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "test-cam"
    assert body["calibrated"] is True
    assert len(body["homography"]) == 3


def test_detect_on_uncalibrated_camera_returns_400(client):
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/detect",
        data={"camera_id": "never-calibrated"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400


def test_calibrate_then_detect_roundtrip(client):
    payload = {
        "camera_id": "test-cam",
        "points": [
            {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
            {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
            {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
            {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
        ],
    }
    client.post("/calibrate", json=payload)

    image_bytes = make_test_image_bytes()
    response = client.post(
        "/detect",
        data={"camera_id": "test-cam", "gap_threshold_m": 4.5},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "test-cam"
    assert len(body["vehicles"]) == 1
    assert body["vehicles"][0]["confidence"] == 0.9


def test_detect_rejects_invalid_image_bytes(client):
    payload = {
        "camera_id": "test-cam",
        "points": [
            {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
            {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
            {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
            {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
        ],
    }
    client.post("/calibrate", json=payload)

    response = client.post(
        "/detect",
        data={"camera_id": "test-cam"},
        files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 422
