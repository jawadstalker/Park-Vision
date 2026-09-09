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
from .parking_detector import detect_parking_zones
from .parking_zones import list_zoned_cameras, load_zones, save_zones
from .schemas import (
    CalibrationRequest,
    CalibrationResponse,
    DetectResponse,
    ParkingSpot,
    ParkingZone,
    Point,
    StatusResponse,
    VehicleDetection,
    ZoneConfigurationRequest,
    ZoneConfigurationResponse,
)

router = APIRouter()


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Invalid image file.")
    return image


@router.post("/zones", response_model=ZoneConfigurationResponse)
async def configure_zones(payload: ZoneConfigurationRequest):
    zones = [zone.model_dump() for zone in payload.zones]
    save_zones(payload.camera_id, zones)
    return ZoneConfigurationResponse(
        camera_id=payload.camera_id,
        saved=True,
        zones=payload.zones,
    )


@router.get("/zones/{camera_id}", response_model=ZoneConfigurationResponse)
async def get_zones(camera_id: str):
    zones = load_zones(camera_id)
    if zones is None:
        raise HTTPException(
            status_code=404,
            detail=f"No parking zones configured for camera '{camera_id}'.",
        )
    parsed = [ParkingZone(**zone) for zone in zones]
    return ZoneConfigurationResponse(camera_id=camera_id, saved=True, zones=parsed)


@router.post("/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    camera_id: str = Form(...),
    confidence: float = Form(0.35),
    image: UploadFile = File(...),
):
    zones = load_zones(camera_id)
    if not zones:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' has no parking zones. Configure /zones first.",
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
    vehicles = detector.detect(image_bgr, conf=confidence)
    spots = detect_parking_zones(vehicles, zones, min_confidence=confidence)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    vehicle_responses = []
    for index, vehicle in enumerate(vehicles, start=1):
        point = vehicle.get("ground_point")
        if point is None:
            x1, _, x2, y2 = vehicle["bbox"]
            point = {"x": (x1 + x2) / 2.0, "y": y2}
        vehicle_responses.append(
            VehicleDetection(
                id=index,
                bbox=vehicle["bbox"],
                confidence=round(vehicle["confidence"], 4),
                ground_point=Point(**point),
                track_id=vehicle.get("track_id"),
            )
        )

    spot_responses = [
        ParkingSpot(
            id=str(spot["id"]),
            row=str(spot["row"]),
            status=spot["status"],
            points=[Point(**point) for point in spot["points"]],
            vehicle_id=spot.get("vehicle_id"),
            area_px=spot.get("area_px", 0.0),
        )
        for spot in spots
    ]

    occupied = sum(spot["status"] == "occupied" for spot in spots)
    empty = len(spots) - occupied

    return DetectResponse(
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
        vehicles=vehicle_responses,
        spots=spot_responses,
        total_spots=len(spots),
        occupied_spots=occupied,
        empty_spots=empty,
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
        cameras_with_parking_zones=list_zoned_cameras(),
        uptime_seconds=round(time.time() - start_time, 2),
    )
