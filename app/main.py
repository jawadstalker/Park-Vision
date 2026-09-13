import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .database import init_db
from .logging_config import configure_logging
from .rate_limit import limiter
from .routes import router
from .vehicle_detector import VehicleDetector

configure_logging()

# Point this at the fine-tuned weights once training is done, e.g.
# "runs/street_parking/weights/best.pt". Falls back to the generic
# pretrained model if the env var isn't set.
MODEL_PATH = os.environ.get("PARK_VISION_MODEL_PATH", "yolov8n.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    init_db()
    app.state.detector = VehicleDetector(model_path=MODEL_PATH, device="cpu")
    yield


app = FastAPI(title="Smart Parking Gap Detection API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(router)
# Versioned alias: existing clients keep using unprefixed paths, new
# integrations should use /v1/... so future breaking changes don't affect them.
app.include_router(router, prefix="/v1")

# Exposes /metrics (request counts, latency histograms, in-progress requests)
# for Prometheus to scrape. Excluded from the /v1 duplication above since
# it's infra, not a versioned API surface.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)