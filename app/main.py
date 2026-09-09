import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import router
from .vehicle_detector import VehicleDetector

# Point this at the fine-tuned weights once training is done, e.g.
# "runs/street_parking/weights/best.pt". Falls back to the generic
# pretrained model if the env var isn't set.
MODEL_PATH = os.environ.get("PARK_VISION_MODEL_PATH", "yolov8n.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    app.state.detector = VehicleDetector(model_path=MODEL_PATH, device="cpu")
    yield


app = FastAPI(title="Smart Parking Gap Detection API", version="0.1.0", lifespan=lifespan)
app.include_router(router)