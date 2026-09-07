import numpy as np

from app.gap_detector import detect_gaps


def test_detect_gaps_finds_empty_spot_above_threshold():
    matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    vehicles = [
        {"bbox": [0, 0, 40, 10]},
        {"bbox": [200, 0, 240, 10]},
    ]
    spots = detect_gaps(matrix, vehicles, threshold_m=4.5)

    statuses = [spot["status"] for spot in spots]
    assert "empty" in statuses
    empty_spot = next(spot for spot in spots if spot["status"] == "empty")
    assert abs(empty_spot["length_m"] - 16.0) < 1e-6


def test_detect_gaps_below_threshold_stays_occupied_only():
    matrix = np.eye(3, dtype=np.float32)
    vehicles = [
        {"bbox": [0, 0, 4, 10]},
        {"bbox": [6, 0, 10, 10]},
    ]
    spots = detect_gaps(matrix, vehicles, threshold_m=4.5)
    assert all(spot["status"] == "occupied" for spot in spots)
    assert len(spots) == 2


def test_detect_gaps_empty_vehicle_list():
    matrix = np.eye(3, dtype=np.float32)
    assert detect_gaps(matrix, [], threshold_m=4.5) == []


def test_detect_gaps_single_vehicle_no_gaps():
    matrix = np.eye(3, dtype=np.float32)
    spots = detect_gaps(matrix, [{"bbox": [0, 0, 10, 10]}], threshold_m=4.5)
    assert len(spots) == 1
    assert spots[0]["status"] == "occupied"


def test_detect_gaps_spots_sorted_by_position():
    matrix = np.eye(3, dtype=np.float32)
    vehicles = [
        {"bbox": [50, 0, 60, 10]},
        {"bbox": [0, 0, 10, 10]},
    ]
    spots = detect_gaps(matrix, vehicles, threshold_m=4.5)
    starts = [spot["start_m"] for spot in spots]
    assert starts == sorted(starts)
