import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import router
from .vehicle_detector import VehicleDetector

MODEL_PATH = os.environ.get("PARK_VISION_MODEL_PATH", "yolov8n.pt")
DEVICE = os.environ.get("PARK_VISION_DEVICE", "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    app.state.detector = VehicleDetector(model_path=MODEL_PATH, device=DEVICE)
    yield


app = FastAPI(
    title="Smart Parking Fixed-Zone Detection API",
    version="0.2.0",
    description="Vehicle detection plus persistent fixed parking-zone occupancy detection.",
    lifespan=lifespan,
)
app.include_router(router)
