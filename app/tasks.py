"""Queue-based inference so multiple cameras don't bottleneck one process.

The synchronous /detect and /detect-pixel endpoints (routes.py) run inline
in the API process and are fine for a single camera or low volume. Once
you're feeding this from many street cameras at once, the API process
becomes the bottleneck: it has to hold a YOLO model in memory and run
inference itself for every request.

This module moves inference into one or more separate worker processes
(`celery -A app.tasks worker`) that pull jobs off a broker (Redis). The API
process (see /detect-async and /jobs/{job_id} in routes.py) only has to
enqueue a job and poll for the result -- it never loads the model.

Run locally:
    redis-server &
    celery -A app.tasks worker --concurrency=2 --loglevel=info

Scale by adding more worker processes/machines; they all pull from the
same Redis queue.
"""
import base64
import os
import time

import cv2
import numpy as np
from celery import Celery

from .calibration import load_calibration
from .database import save_detection
from .gap_detector import DEFAULT_GAP_THRESHOLD_M, detect_gaps
from .vehicle_detector import VehicleDetector

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)
MODEL_PATH = os.environ.get("PARK_VISION_MODEL_PATH", "yolov8n.pt")

celery_app = Celery("park_vision", broker=REDIS_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # Overridable in tests (see tests/test_tasks.py) so the suite doesn't
    # need a real Redis broker: tasks execute inline instead of via a worker.
    task_always_eager=os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true",
)

_worker_detector = None


def _get_detector() -> VehicleDetector:
    """Lazily loads the model once per worker process, not per task."""
    global _worker_detector
    if _worker_detector is None:
        _worker_detector = VehicleDetector(model_path=MODEL_PATH, device="cpu")
    return _worker_detector


@celery_app.task(name="park_vision.run_detection", bind=True, max_retries=2)
def run_detection_task(self, image_b64: str, camera_id: str, gap_threshold_m: float = DEFAULT_GAP_THRESHOLD_M):
    matrix = load_calibration(camera_id)
    if matrix is None:
        return {"error": f"Camera '{camera_id}' is not calibrated. Call /calibrate first."}

    image_bytes = base64.b64decode(image_b64)
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"error": "Invalid image file."}

    height, width = image_bgr.shape[:2]
    detector = _get_detector()

    start_time = time.perf_counter()
    vehicles = detector.detect(image_bgr)
    spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold_m)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    save_detection(
        camera_id=camera_id,
        mode="calibrated",
        frame_width=width,
        frame_height=height,
        vehicles=vehicles,
        spots=spots,
        processing_time_ms=round(elapsed_ms, 2),
    )

    return {
        "camera_id": camera_id,
        "frame_width": width,
        "frame_height": height,
        "gap_threshold_m": gap_threshold_m,
        "vehicles": [
            {
                "id": i + 1,
                "bbox": v["bbox"],
                "confidence": round(v["confidence"], 4),
                "ground_point": {"x": (v["bbox"][0] + v["bbox"][2]) / 2.0, "y": v["bbox"][3]},
            }
            for i, v in enumerate(vehicles)
        ],
        "spots": [
            {
                "id": s["id"],
                "status": s["status"],
                "start_m": s["start_m"],
                "end_m": s["end_m"],
                "length_m": s["length_m"],
                "vehicle_id": None,
            }
            for s in spots
        ],
        "processing_time_ms": round(elapsed_ms, 2),
    }
