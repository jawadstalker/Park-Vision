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


class FakeTwoVehicleDetector(FakeDetector):
    def detect(self, image_bgr, conf=0.35):
        return [
            {"bbox": [20.0, 190.0, 110.0, 250.0], "confidence": 0.9},
            {"bbox": [420.0, 190.0, 510.0, 250.0], "confidence": 0.85},
        ]


@pytest.fixture
def two_vehicle_client(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeTwoVehicleDetector)
    with TestClient(main_module.app) as test_client:
        yield test_client


def make_wide_test_image_bytes():
    image = Image.new("RGB", (900, 300), color=(200, 200, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_detect_pixel_returns_vehicles_and_spots(two_vehicle_client):
    image_bytes = make_wide_test_image_bytes()
    response = two_vehicle_client.post(
        "/detect-pixel",
        data={"conf": 0.35, "enable_recheck": "false"},
        files={"image": ("wide.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["vehicles"]) == 2
    statuses = {spot["status"] for spot in body["spots"]}
    assert "occupied" in statuses
    assert "empty" in statuses


def test_detect_pixel_without_camera_id_does_not_save_history(two_vehicle_client):
    image_bytes = make_wide_test_image_bytes()
    two_vehicle_client.post(
        "/detect-pixel",
        data={"enable_recheck": "false"},
        files={"image": ("wide.jpg", image_bytes, "image/jpeg")},
    )
    response = two_vehicle_client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_detect_pixel_with_camera_id_saves_history(two_vehicle_client):
    image_bytes = make_wide_test_image_bytes()
    two_vehicle_client.post(
        "/detect-pixel",
        data={"camera_id": "cam-pixel", "enable_recheck": "false"},
        files={"image": ("wide.jpg", image_bytes, "image/jpeg")},
    )
    response = two_vehicle_client.get("/history/cam-pixel")
    assert response.status_code == 200
    body = response.json()
    assert len(body["records"]) == 1
    assert body["records"][0]["mode"] == "pixel"


def test_detect_calibrated_endpoint_also_saves_history(client):
    payload = {
        "camera_id": "hist-cam",
        "points": [
            {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
            {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
            {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
            {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
        ],
    }
    client.post("/calibrate", json=payload)
    image_bytes = make_test_image_bytes()
    client.post(
        "/detect",
        data={"camera_id": "hist-cam"},
        files={"image": ("t.jpg", image_bytes, "image/jpeg")},
    )
    response = client.get("/history/hist-cam")
    assert response.status_code == 200
    body = response.json()
    assert len(body["records"]) == 1
    assert body["records"][0]["mode"] == "calibrated"


def test_history_unknown_camera_returns_empty_list(client):
    response = client.get("/history/never-seen")
    assert response.status_code == 200
    assert response.json()["records"] == []
