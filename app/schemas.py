from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class VehicleDetection(BaseModel):
    id: int
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float
    ground_point: Point


class ParkingSpot(BaseModel):
    id: int
    status: Literal["occupied", "empty"]
    start_m: float
    end_m: float
    length_m: float
    vehicle_id: Optional[int] = None


class DetectResponse(BaseModel):
    camera_id: str
    frame_width: int
    frame_height: int
    gap_threshold_m: float
    vehicles: List[VehicleDetection]
    spots: List[ParkingSpot]
    processing_time_ms: float


class CalibrationPoint(BaseModel):
    pixel: Point
    real_world: Point


class CalibrationRequest(BaseModel):
    camera_id: str
    points: List[CalibrationPoint] = Field(..., min_length=4, max_length=4)


class CalibrationResponse(BaseModel):
    camera_id: str
    calibrated: bool
    homography: List[List[float]]


class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    cameras_calibrated: List[str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
