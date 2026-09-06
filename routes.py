import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .calibration import (
    compute_homography,
    list_calibrated_cameras,
    load_calibration,
    save_calibration,
)
from .gap_detector import DEFAULT_GAP_THRESHOLD_M, detect_gaps
from .schemas import (
    CalibrationRequest,
    CalibrationResponse,
    DetectResponse,
    ParkingSpot,
    Point,
    StatusResponse,
    VehicleDetection,
)

router = APIRouter()


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Invalid image file.")
    return image


@router.post("/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    camera_id: str = Form(...),
    gap_threshold_m: float = Form(DEFAULT_GAP_THRESHOLD_M),
    image: UploadFile = File(...),
):
    matrix = load_calibration(camera_id)
    if matrix is None:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' is not calibrated. Call /calibrate first.",
        )

    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    image_bgr = decode_image(image_bytes)
    height, width = image_bgr.shape[:2]

    start_time = time.perf_counter()
    vehicles = detector.detect(image_bgr)
    spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold_m)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    vehicle_responses = [
        VehicleDetection(
            id=i + 1,
            bbox=v["bbox"],
            confidence=round(v["confidence"], 4),
            ground_point=Point(
                x=(v["bbox"][0] + v["bbox"][2]) / 2.0,
                y=v["bbox"][3],
            ),
        )
        for i, v in enumerate(vehicles)
    ]

    spot_responses = [
        ParkingSpot(
            id=s["id"],
            status=s["status"],
            start_m=s["start_m"],
            end_m=s["end_m"],
            length_m=s["length_m"],
            vehicle_id=None,
        )
        for s in spots
    ]

    return DetectResponse(
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
        gap_threshold_m=gap_threshold_m,
        vehicles=vehicle_responses,
        spots=spot_responses,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(payload: CalibrationRequest):
    pixel_points = [(p.pixel.x, p.pixel.y) for p in payload.points]
    real_world_points = [(p.real_world.x, p.real_world.y) for p in payload.points]
    matrix = compute_homography(pixel_points, real_world_points)
    save_calibration(payload.camera_id, matrix)
    return CalibrationResponse(
        camera_id=payload.camera_id,
        calibrated=True,
        homography=matrix.tolist(),
    )


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    detector = getattr(request.app.state, "detector", None)
    start_time = getattr(request.app.state, "start_time", time.time())
    return StatusResponse(
        model_loaded=detector is not None,
        device=detector.device if detector is not None else "none",
        cameras_calibrated=list_calibrated_cameras(),
        uptime_seconds=round(time.time() - start_time, 2),
    )
