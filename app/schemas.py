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
    track_id: Optional[int] = None


class ParkingZonePoint(BaseModel):
    x: float
    y: float


class ParkingZone(BaseModel):
    id: str
    row: str = "A"
    points: List[ParkingZonePoint] = Field(..., min_length=3)


class ParkingSpot(BaseModel):
    id: str
    row: str
    status: Literal["occupied", "empty"]
    points: List[ParkingZonePoint]
    vehicle_id: Optional[int] = None
    area_px: float = 0.0


class DetectResponse(BaseModel):
    camera_id: str
    frame_width: int
    frame_height: int
    vehicles: List[VehicleDetection]
    spots: List[ParkingSpot]
    total_spots: int
    occupied_spots: int
    empty_spots: int
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


class ZoneConfigurationRequest(BaseModel):
    camera_id: str
    zones: List[ParkingZone] = Field(..., min_length=1)


class ZoneConfigurationResponse(BaseModel):
    camera_id: str
    saved: bool
    zones: List[ParkingZone]


class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    cameras_calibrated: List[str]
    cameras_with_parking_zones: List[str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
