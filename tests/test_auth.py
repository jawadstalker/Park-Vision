import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app import auth as auth_module


class FakeDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = device

    def detect(self, image_bgr, conf=0.35):
        return [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9}]


@pytest.fixture
def client_with_keys(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    monkeypatch.setenv(auth_module.API_KEY_ENV_VAR, "secret-key-1:*;secret-key-2:*")
    with TestClient(main_module.app) as test_client:
        yield test_client


def make_test_image_bytes():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_calibrate_without_key_is_rejected(client_with_keys):
    response = client_with_keys.post(
        "/calibrate",
        json={
            "camera_id": "cam",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 401


def test_calibrate_with_wrong_key_is_rejected(client_with_keys):
    response = client_with_keys.post(
        "/calibrate",
        headers={"X-API-Key": "not-a-real-key"},
        json={
            "camera_id": "cam",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 401


def test_calibrate_with_valid_key_succeeds(client_with_keys):
    response = client_with_keys.post(
        "/calibrate",
        headers={"X-API-Key": "secret-key-2"},
        json={
            "camera_id": "cam",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 200


def test_detect_without_key_is_rejected(client_with_keys):
    image_bytes = make_test_image_bytes()
    response = client_with_keys.post(
        "/detect",
        data={"camera_id": "never-calibrated"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 401


def test_status_stays_public_even_with_keys_configured(client_with_keys):
    response = client_with_keys.get("/status")
    assert response.status_code == 200


def test_auth_disabled_when_no_keys_configured(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    monkeypatch.delenv(auth_module.API_KEY_ENV_VAR, raising=False)
    with TestClient(main_module.app) as test_client:
        response = test_client.post(
            "/calibrate",
            json={
                "camera_id": "cam",
                "points": [
                    {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                    {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                    {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                    {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
                ],
            },
        )
        assert response.status_code == 200


@pytest.fixture
def scoped_client(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    monkeypatch.setenv(
        auth_module.API_KEY_ENV_VAR, "key-cam-a:cam-a,cam-b;key-all:*"
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_scoped_key_can_calibrate_its_own_camera(scoped_client):
    response = scoped_client.post(
        "/calibrate",
        headers={"X-API-Key": "key-cam-a"},
        json={
            "camera_id": "cam-a",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 200


def test_scoped_key_cannot_calibrate_other_camera(scoped_client):
    response = scoped_client.post(
        "/calibrate",
        headers={"X-API-Key": "key-cam-a"},
        json={
            "camera_id": "cam-z",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 403


def test_wildcard_key_can_access_any_camera(scoped_client):
    response = scoped_client.post(
        "/calibrate",
        headers={"X-API-Key": "key-all"},
        json={
            "camera_id": "cam-anything",
            "points": [
                {"pixel": {"x": 0, "y": 100}, "real_world": {"x": 0, "y": 0}},
                {"pixel": {"x": 100, "y": 100}, "real_world": {"x": 10, "y": 0}},
                {"pixel": {"x": 100, "y": 0}, "real_world": {"x": 10, "y": 10}},
                {"pixel": {"x": 0, "y": 0}, "real_world": {"x": 0, "y": 10}},
            ],
        },
    )
    assert response.status_code == 200
