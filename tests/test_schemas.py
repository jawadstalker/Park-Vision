import pytest
from pydantic import ValidationError

from app.schemas import CalibrationPoint, CalibrationRequest, ParkingSpot, Point, VehicleDetection


def test_vehicle_detection_valid():
    vehicle = VehicleDetection(id=1, bbox=[0, 0, 10, 10], confidence=0.9, ground_point=Point(x=5, y=10))
    assert vehicle.id == 1
    assert vehicle.bbox == [0, 0, 10, 10]


def test_vehicle_detection_rejects_wrong_length_bbox():
    with pytest.raises(ValidationError):
        VehicleDetection(id=1, bbox=[0, 0, 10], confidence=0.9, ground_point=Point(x=5, y=10))


def test_parking_spot_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ParkingSpot(id=1, status="unknown", start_m=0, end_m=1, length_m=1)


def test_parking_spot_accepts_valid_status():
    spot = ParkingSpot(id=1, status="empty", start_m=0, end_m=5, length_m=5)
    assert spot.status == "empty"


def test_calibration_request_requires_exactly_four_points():
    point = CalibrationPoint(pixel=Point(x=0, y=0), real_world=Point(x=0, y=0))
    with pytest.raises(ValidationError):
        CalibrationRequest(camera_id="cam", points=[point, point, point])


def test_calibration_request_accepts_four_points():
    point = CalibrationPoint(pixel=Point(x=0, y=0), real_world=Point(x=0, y=0))
    request = CalibrationRequest(camera_id="cam", points=[point, point, point, point])
    assert len(request.points) == 4
