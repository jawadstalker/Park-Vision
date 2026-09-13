import base64
import logging
import os
import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .auth import authorize_camera_access, require_api_key
from .calibration import (
    BadCalibrationError,
    compute_homography,
    list_calibrated_cameras,
    load_calibration,
    save_calibration,
)
from .database import get_history, list_cameras_with_history, save_detection
from .gap_detector import DEFAULT_GAP_THRESHOLD_M, detect_gaps
from .pixel_gap_detector import (
    DEFAULT_MIN_GAP_RATIO,
    DEFAULT_ROW_TOLERANCE_RATIO,
    detect_pixel_gaps_multi_row,
    refine_with_low_confidence_recheck,
)
from .rate_limit import DETECT_RATE_LIMIT, limiter
from .schemas import (
    CalibrationRequest,
    CalibrationResponse,
    DetectResponse,
    HistoryResponse,
    JobEnqueuedResponse,
    JobStatusResponse,
    ParkingSpot,
    PixelDetectResponse,
    PixelSpot,
    PixelVehicleDetection,
    Point,
    StatusResponse,
    VehicleDetection,
)
from .tasks import celery_app, run_detection_task

router = APIRouter()
logger = logging.getLogger("park_vision.routes")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Invalid image file.")
    return image


async def read_upload_within_limit(image: UploadFile) -> bytes:
    """Reads the upload but refuses anything over MAX_UPLOAD_MB instead of
    buffering an arbitrarily large file into memory (a single oversized
    request could otherwise take a worker down)."""
    chunks = []
    total = 0
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/detect", response_model=DetectResponse)
@limiter.limit(DETECT_RATE_LIMIT)
async def detect(
    request: Request,
    camera_id: str = Form(...),
    gap_threshold_m: float = Form(DEFAULT_GAP_THRESHOLD_M),
    image: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):
    authorize_camera_access(api_key, camera_id)
    image_bytes = await read_upload_within_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    matrix = load_calibration(camera_id)
    if matrix is None:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' is not calibrated. Call /calibrate first.",
        )

    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bgr = decode_image(image_bytes)
    height, width = image_bgr.shape[:2]

    start_time = time.perf_counter()
    vehicles = await run_in_threadpool(detector.detect, image_bgr)
    spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold_m)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "detect completed",
        extra={"camera_id": camera_id, "duration_ms": round(elapsed_ms, 2), "path": "/detect"},
    )
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

    save_detection(
        camera_id=camera_id,
        mode="calibrated",
        frame_width=width,
        frame_height=height,
        vehicles=vehicles,
        spots=spots,
        processing_time_ms=round(elapsed_ms, 2),
    )

    return DetectResponse(
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
        gap_threshold_m=gap_threshold_m,
        vehicles=vehicle_responses,
        spots=spot_responses,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/detect-pixel", response_model=PixelDetectResponse)
@limiter.limit(DETECT_RATE_LIMIT)
async def detect_pixel(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form(None),
    conf: float = Form(0.35),
    min_gap_ratio: float = Form(DEFAULT_MIN_GAP_RATIO),
    row_tolerance_ratio: float = Form(DEFAULT_ROW_TOLERANCE_RATIO),
    enable_recheck: bool = Form(True),
    api_key: str = Depends(require_api_key),
):
    if camera_id:
        authorize_camera_access(api_key, camera_id)
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bytes = await read_upload_within_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    image_bgr = decode_image(image_bytes)
    height, width = image_bgr.shape[:2]

    start_time = time.perf_counter()
    vehicles = await run_in_threadpool(detector.detect, image_bgr, conf=conf)
    if enable_recheck and len(vehicles) >= 2:
        vehicles = await run_in_threadpool(
            refine_with_low_confidence_recheck, detector, image_bgr, vehicles
        )
    slots = detect_pixel_gaps_multi_row(
        vehicles,
        min_gap_ratio=min_gap_ratio,
        row_tolerance_ratio=row_tolerance_ratio,
        frame_width=width,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    vehicle_responses = [
        PixelVehicleDetection(id=i + 1, bbox=v["bbox"], confidence=round(v["confidence"], 4))
        for i, v in enumerate(vehicles)
    ]
    spot_responses = [
        PixelSpot(id=s["id"], status=s["status"], bbox=s["bbox"]) for s in slots
    ]

    if camera_id:
        save_detection(
            camera_id=camera_id,
            mode="pixel",
            frame_width=width,
            frame_height=height,
            vehicles=vehicles,
            spots=slots,
            processing_time_ms=round(elapsed_ms, 2),
        )

    return PixelDetectResponse(
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
        recheck_enabled=enable_recheck,
        vehicles=vehicle_responses,
        spots=spot_responses,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.get("/history/{camera_id}", response_model=HistoryResponse)
async def history(camera_id: str, limit: int = 50):
    records = get_history(camera_id, limit=limit)
    return HistoryResponse(camera_id=camera_id, records=records)


@router.get("/history", response_model=List[str])
async def history_cameras():
    return list_cameras_with_history()


@router.post("/calibrate", response_model=CalibrationResponse)
@limiter.limit(DETECT_RATE_LIMIT)
async def calibrate(
    request: Request, payload: CalibrationRequest, api_key: str = Depends(require_api_key)
):
    authorize_camera_access(api_key, payload.camera_id)
    pixel_points = [(p.pixel.x, p.pixel.y) for p in payload.points]
    real_world_points = [(p.real_world.x, p.real_world.y) for p in payload.points]
    try:
        matrix = compute_homography(pixel_points, real_world_points)
    except BadCalibrationError as exc:
        logger.warning(
            "calibration rejected",
            extra={"camera_id": payload.camera_id, "path": "/calibrate"},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    save_calibration(payload.camera_id, matrix)
    return CalibrationResponse(
        camera_id=payload.camera_id,
        calibrated=True,
        homography=matrix.tolist(),
    )


@router.post("/detect-async", response_model=JobEnqueuedResponse)
@limiter.limit(DETECT_RATE_LIMIT)
async def detect_async(
    request: Request,
    camera_id: str = Form(...),
    gap_threshold_m: float = Form(DEFAULT_GAP_THRESHOLD_M),
    image: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):
    """Enqueues a detection job on a Celery worker instead of running inline.

    Use this instead of /detect when running many cameras: it lets the API
    process stay lightweight (no model loaded, never blocked on inference)
    while one or more `celery -A app.tasks worker` processes do the work.
    Poll /jobs/{job_id} for the result.
    """
    authorize_camera_access(api_key, camera_id)
    image_bytes = await read_upload_within_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if load_calibration(camera_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' is not calibrated. Call /calibrate first.",
        )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    task = run_detection_task.delay(image_b64, camera_id, gap_threshold_m)
    return JobEnqueuedResponse(job_id=task.id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    result = celery_app.AsyncResult(job_id)
    payload = None
    error = None
    if result.successful():
        value = result.result
        if isinstance(value, dict) and "error" in value:
            error = value["error"]
        else:
            payload = value
    elif result.failed():
        error = str(result.result)

    return JobStatusResponse(
        job_id=job_id,
        state=result.state,
        result=payload,
        error=error,
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
