import numpy as np

from app.gap_detector import detect_gaps, split_gap_into_spots


def test_detect_gaps_small_gap_becomes_one_spot():
    matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    vehicles = [
        {"bbox": [0, 0, 40, 10]},
        {"bbox": [130, 0, 170, 10]},
    ]
    spots = detect_gaps(matrix, vehicles, threshold_m=4.5, spot_length_m=5.0)

    empty_spots = [spot for spot in spots if spot["status"] == "empty"]
    assert len(empty_spots) == 1
    assert abs(empty_spots[0]["length_m"] - 5.0) < 1e-6


def test_detect_gaps_large_gap_splits_into_multiple_spots():
    # A 236 m gap must become many discrete spots, not one giant blob.
    matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    vehicles = [
        {"bbox": [0, 0, 40, 10]},
        {"bbox": [2400, 0, 2440, 10]},
    ]
    spots = detect_gaps(matrix, vehicles, threshold_m=4.5, spot_length_m=5.0)

    empty_spots = [spot for spot in spots if spot["status"] == "empty"]
    assert len(empty_spots) == 47
    for spot in empty_spots:
        assert abs(spot["length_m"] - 5.0) < 1e-6
    # Spots must not overlap and must stay within the original gap bounds.
    for a, b in zip(empty_spots, empty_spots[1:]):
        assert a["end_m"] <= b["start_m"] + 1e-6


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


def test_split_gap_into_spots_centers_leftover_margin():
    spots = split_gap_into_spots(gap_start=0.0, gap_end=12.0, spot_length_m=5.0)
    assert len(spots) == 2
    assert abs(spots[0]["start_m"] - 1.0) < 1e-6
    assert abs(spots[-1]["end_m"] - 11.0) < 1e-6


def test_split_gap_into_spots_gap_smaller_than_spot_length_still_returns_one():
    spots = split_gap_into_spots(gap_start=0.0, gap_end=3.0, spot_length_m=5.0)
    assert len(spots) == 1