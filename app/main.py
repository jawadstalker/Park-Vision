import time

from fastapi import FastAPI

from .routes import router
from .vehicle_detector import VehicleDetector

app = FastAPI(title="Smart Parking Gap Detection API", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    app.state.start_time = time.time()
    app.state.detector = VehicleDetector(model_path="yolov8n.pt", device="cpu")


app.include_router(router)
