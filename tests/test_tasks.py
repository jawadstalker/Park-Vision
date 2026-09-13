import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
import app.tasks as tasks_module
from app.calibration import compute_homography, save_calibration


class FakeDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = device

    def detect(self, image_bgr, conf=0.35):
        return [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9}]


@pytest.fixture(autouse=True)
def eager_celery(monkeypatch):
    # Run tasks inline for tests instead of needing a real Redis broker/worker,
    # and store results in-memory so AsyncResult lookups don't hit Redis either.
    monkeypatch.setattr(tasks_module.celery_app.conf, "task_always_eager", True)
    monkeypatch.setattr(tasks_module.celery_app.conf, "task_store_eager_result", True)
    monkeypatch.setattr(tasks_module.celery_app.conf, "result_backend", "cache+memory://")
    monkeypatch.setattr(tasks_module.celery_app, "_backend_cache", None)
    tasks_module.celery_app._local.backend = None
    monkeypatch.setattr(tasks_module, "_worker_detector", FakeDetector())
    yield


def make_test_image_bytes():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_run_detection_task_returns_error_for_uncalibrated_camera():
    image_b64 = base64.b64encode(make_test_image_bytes()).decode("ascii")
    result = tasks_module.run_detection_task.apply(
        args=(image_b64, "never-calibrated", 4.5)
    ).get()
    assert "error" in result


def test_run_detection_task_returns_detections_for_calibrated_camera():
    matrix = compute_homography(
        [(0, 480), (640, 480), (640, 300), (0, 300)],
        [(0, 0), (20, 0), (20, 6), (0, 6)],
    )
    save_calibration("task-cam", matrix)

    image_b64 = base64.b64encode(make_test_image_bytes()).decode("ascii")
    result = tasks_module.run_detection_task.apply(args=(image_b64, "task-cam", 4.5)).get()

    assert result["camera_id"] == "task-cam"
    assert len(result["vehicles"]) == 1
    assert result["vehicles"][0]["confidence"] == 0.9


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_detect_async_then_poll_job_status(client):
    matrix = compute_homography(
        [(0, 480), (640, 480), (640, 300), (0, 300)],
        [(0, 0), (20, 0), (20, 6), (0, 6)],
    )
    save_calibration("async-cam", matrix)

    image_bytes = make_test_image_bytes()
    enqueue_response = client.post(
        "/detect-async",
        data={"camera_id": "async-cam"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert enqueue_response.status_code == 200
    job_id = enqueue_response.json()["job_id"]

    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["state"] == "SUCCESS"
    assert body["result"]["camera_id"] == "async-cam"
    assert len(body["result"]["vehicles"]) == 1


def test_detect_async_rejects_uncalibrated_camera_before_enqueue(client):
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/detect-async",
        data={"camera_id": "never-calibrated"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400


def test_detect_async_requires_api_key(client, monkeypatch):
    from app import auth as auth_module

    monkeypatch.setenv(auth_module.API_KEY_ENV_VAR, "secret")
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/detect-async",
        data={"camera_id": "any-cam"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 401
