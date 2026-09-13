import io

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import app.main as main_module
import app.routes as routes_module
from app.rate_limit import limiter, rate_limit_key


class FakeDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = device

    def detect(self, image_bgr, conf=0.35):
        return []


def make_test_image_bytes():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_rate_limit_key_prefers_api_key_header():
    class FakeRequest:
        headers = {"X-API-Key": "abc"}
        client = type("c", (), {"host": "1.2.3.4"})()

    assert rate_limit_key(FakeRequest()) == "abc"


def test_rate_limit_key_falls_back_to_ip():
    class FakeRequest:
        headers = {}
        client = type("c", (), {"host": "1.2.3.4"})()

    assert rate_limit_key(FakeRequest()) == "1.2.3.4"


@pytest.fixture
def toy_limited_app():
    # Exercises the *real* shared limiter instance against a throwaway
    # route with a very low limit, so the test doesn't have to cripple
    # the production endpoints' actual rate limit to be fast/deterministic.
    limiter.reset()
    toy_app = FastAPI()
    toy_app.state.limiter = limiter
    toy_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @toy_app.get("/ping")
    @limiter.limit("2/minute")
    async def ping(request: Request):
        return {"ok": True}

    with TestClient(toy_app) as client:
        yield client
    limiter.reset()


def test_requests_within_limit_succeed(toy_limited_app):
    assert toy_limited_app.get("/ping").status_code == 200
    assert toy_limited_app.get("/ping").status_code == 200


def test_requests_over_limit_are_rejected(toy_limited_app):
    toy_limited_app.get("/ping")
    toy_limited_app.get("/ping")
    response = toy_limited_app.get("/ping")
    assert response.status_code == 429


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "VehicleDetector", FakeDetector)
    limiter.reset()
    with TestClient(main_module.app) as test_client:
        yield test_client
    limiter.reset()


def test_detect_rejects_upload_over_size_limit(client, monkeypatch):
    monkeypatch.setattr(routes_module, "MAX_UPLOAD_BYTES", 10)
    image_bytes = make_test_image_bytes()
    assert len(image_bytes) > 10
    response = client.post(
        "/detect",
        data={"camera_id": "any-cam"},
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 413


def test_v1_prefix_mirrors_unprefixed_status(client):
    unprefixed = client.get("/status")
    versioned = client.get("/v1/status")
    assert unprefixed.status_code == versioned.status_code == 200
    assert unprefixed.json()["device"] == versioned.json()["device"]


def test_metrics_endpoint_is_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests" in response.content or b"# HELP" in response.content
