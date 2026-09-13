from typing import Any, Dict, List, Literal, Optional
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


class PixelVehicleDetection(BaseModel):
    id: int
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float


class PixelSpot(BaseModel):
    id: int
    status: Literal["occupied", "empty"]
    bbox: List[float] = Field(..., min_length=4, max_length=4)


class PixelDetectResponse(BaseModel):
    camera_id: Optional[str] = None
    frame_width: int
    frame_height: int
    recheck_enabled: bool
    vehicles: List[PixelVehicleDetection]
    spots: List[PixelSpot]
    processing_time_ms: float


class HistoryRecord(BaseModel):
    id: int
    camera_id: str
    mode: str
    timestamp: str
    frame_width: int
    frame_height: int
    vehicle_count: int
    occupied_count: int
    empty_count: int
    processing_time_ms: float
    vehicles: List[Dict[str, Any]]
    spots: List[Dict[str, Any]]


class HistoryResponse(BaseModel):
    camera_id: str
    records: List[HistoryRecord]


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


class JobEnqueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
