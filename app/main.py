import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import router
from .vehicle_detector import VehicleDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    app.state.detector = VehicleDetector(model_path="yolov8n.pt", device="cpu")
    yield


app = FastAPI(title="Smart Parking Gap Detection API", version="0.1.0", lifespan=lifespan)
app.include_router(router)
