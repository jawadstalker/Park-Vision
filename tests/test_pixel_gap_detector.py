from app.pixel_gap_detector import (
    cluster_vehicles_by_row,
    detect_pixel_gaps,
    detect_pixel_gaps_multi_row,
    refine_with_low_confidence_recheck,
)


def test_detect_pixel_gaps_empty_vehicle_list():
    assert detect_pixel_gaps([]) == []


def test_detect_pixel_gaps_single_vehicle_no_gaps():
    slots = detect_pixel_gaps([{"bbox": [0, 0, 100, 50]}])
    assert len(slots) == 1
    assert slots[0]["status"] == "occupied"


def test_detect_pixel_gaps_small_gap_becomes_one_spot():
    vehicles = [
        {"bbox": [0, 0, 100, 50]},
        {"bbox": [190, 0, 290, 50]},
    ]
    slots = detect_pixel_gaps(vehicles, min_gap_ratio=0.85)
    empty = [s for s in slots if s["status"] == "empty"]
    assert len(empty) == 1
    assert abs((empty[0]["bbox"][2] - empty[0]["bbox"][0]) - 100) < 1e-6


def test_detect_pixel_gaps_uses_local_width_not_global_average():
    # A near (wide), a mid, and a far (narrow) vehicle. The gap between the
    # near and mid car must use their local average width, not a global
    # average dragged down by the far car's much smaller width.
    vehicles = [
        {"bbox": [20, 200, 170, 280]},   # width 150 (near)
        {"bbox": [300, 215, 400, 270]},  # width 100 (mid)
        {"bbox": [600, 225, 660, 265]},  # width 60 (far)
    ]
    slots = detect_pixel_gaps(vehicles, min_gap_ratio=0.85)
    empty = [s for s in slots if s["status"] == "empty"]

    near_gap_spot = min(empty, key=lambda s: s["bbox"][0])
    far_gap_spot = max(empty, key=lambda s: s["bbox"][0])

    near_width = near_gap_spot["bbox"][2] - near_gap_spot["bbox"][0]
    far_width = far_gap_spot["bbox"][2] - far_gap_spot["bbox"][0]

    assert abs(near_width - 125.0) < 1e-6
    assert abs(far_width - 80.0) < 1e-6
    assert near_width > far_width


def test_detect_pixel_gaps_interpolates_vertical_position():
    vehicles = [
        {"bbox": [0, 200, 100, 280]},
        {"bbox": [400, 240, 460, 280]},
    ]
    slots = detect_pixel_gaps(vehicles, min_gap_ratio=0.85)
    empty = [s for s in slots if s["status"] == "empty"]
    assert len(empty) >= 1
    # The interpolated top must sit strictly between the two vehicles' tops.
    for spot in empty:
        assert 200 <= spot["bbox"][1] <= 240


def test_detect_pixel_gaps_ids_assigned_in_left_to_right_order():
    vehicles = [
        {"bbox": [300, 0, 400, 50]},
        {"bbox": [0, 0, 100, 50]},
    ]
    slots = detect_pixel_gaps(vehicles)
    xs = [s["bbox"][0] for s in slots]
    assert xs == sorted(xs)
    assert [s["id"] for s in slots] == list(range(1, len(slots) + 1))


def test_cluster_vehicles_by_row_separates_two_rows():
    vehicles = [
        {"bbox": [20, 50, 100, 90]},
        {"bbox": [300, 55, 380, 95]},
        {"bbox": [50, 200, 160, 270]},
        {"bbox": [500, 210, 610, 280]},
    ]
    rows = cluster_vehicles_by_row(vehicles)
    assert len(rows) == 2
    assert {len(row) for row in rows} == {2}


def test_cluster_vehicles_by_row_single_row_stays_together():
    vehicles = [
        {"bbox": [0, 100, 80, 180]},
        {"bbox": [200, 105, 280, 185]},
        {"bbox": [400, 95, 480, 175]},
    ]
    rows = cluster_vehicles_by_row(vehicles)
    assert len(rows) == 1
    assert len(rows[0]) == 3


def test_detect_pixel_gaps_multi_row_does_not_mix_rows():
    vehicles = [
        {"bbox": [20, 50, 100, 90]},
        {"bbox": [300, 55, 380, 95]},
        {"bbox": [50, 200, 160, 270]},
        {"bbox": [500, 210, 610, 280]},
    ]
    slots = detect_pixel_gaps_multi_row(vehicles)
    empty_spots = [s for s in slots if s["status"] == "empty"]
    # No empty spot should span across the vertical gap between the two rows.
    for spot in empty_spots:
        assert spot["bbox"][1] < 150 or spot["bbox"][1] > 150


class _FakeDetectorFindsOneMissedCarAtLowConfidence:
    def detect(self, image_bgr, conf=0.35):
        if conf <= 0.2:
            h, w = image_bgr.shape[:2]
            return [{"bbox": [10, 10, w - 10, h - 10], "confidence": 0.18}]
        return []


class _FakeDetectorReturnsDuplicateBoxesForSameCar:
    def detect(self, image_bgr, conf=0.35):
        if conf <= 0.2:
            return [
                {"bbox": [90, 110, 390, 170], "confidence": 0.18},
                {"bbox": [92, 112, 388, 168], "confidence": 0.16},
            ]
        return []


def test_refine_with_low_confidence_recheck_finds_missed_vehicle():
    import numpy as np

    vehicles = [
        {"bbox": [0, 100, 80, 180]},
        {"bbox": [400, 100, 480, 180]},
    ]
    fake_image = np.zeros((300, 500, 3), dtype=np.uint8)

    augmented = refine_with_low_confidence_recheck(
        _FakeDetectorFindsOneMissedCarAtLowConfidence(), fake_image, vehicles, recheck_conf=0.15
    )
    assert len(augmented) == 3


def test_refine_with_low_confidence_recheck_dedupes_overlapping_boxes():
    import numpy as np

    vehicles = [
        {"bbox": [0, 100, 80, 180]},
        {"bbox": [400, 100, 480, 180]},
    ]
    fake_image = np.zeros((300, 500, 3), dtype=np.uint8)

    augmented = refine_with_low_confidence_recheck(
        _FakeDetectorReturnsDuplicateBoxesForSameCar(),
        fake_image,
        vehicles,
        recheck_conf=0.15,
        dedupe_iou=0.3,
    )
    assert len(augmented) == 3


def test_refine_with_low_confidence_recheck_skips_small_gaps():
    import numpy as np

    class _ShouldNeverBeCalled:
        def detect(self, image_bgr, conf=0.35):
            raise AssertionError("recheck should not run for a small gap")

    vehicles = [
        {"bbox": [0, 100, 80, 180]},
        {"bbox": [90, 100, 170, 180]},
    ]
    fake_image = np.zeros((300, 500, 3), dtype=np.uint8)

    augmented = refine_with_low_confidence_recheck(_ShouldNeverBeCalled(), fake_image, vehicles)
    assert len(augmented) == 2


def test_detect_pixel_gaps_single_vehicle_without_frame_width_stays_zero_empty():
    vehicles = [{"bbox": [400, 200, 500, 260]}]
    slots = detect_pixel_gaps(vehicles)
    assert sum(1 for s in slots if s["status"] == "empty") == 0


def test_detect_pixel_gaps_single_vehicle_with_frame_width_finds_edge_space():
    vehicles = [{"bbox": [400, 200, 500, 260]}]
    slots = detect_pixel_gaps(vehicles, frame_width=900)
    empty = [s for s in slots if s["status"] == "empty"]
    assert len(empty) == 8
    # Space strictly before and after the single vehicle must both be covered.
    assert any(s["bbox"][1] < 400 for s in empty) or any(s["bbox"][0] < 400 for s in empty)
    assert any(s["bbox"][0] >= 500 for s in empty)


def test_detect_pixel_gaps_edge_space_too_small_is_ignored():
    vehicles = [{"bbox": [5, 200, 105, 260]}]  # only 5px before it, way less than one car width
    slots = detect_pixel_gaps(vehicles, frame_width=110)
    empty = [s for s in slots if s["status"] == "empty"]
    assert len(empty) == 0


def test_detect_pixel_gaps_multi_row_passes_frame_width_through():
    vehicles = [
        {"bbox": [400, 50, 500, 90]},
        {"bbox": [400, 200, 500, 260]},
    ]
    slots = detect_pixel_gaps_multi_row(vehicles, frame_width=900)
    empty = [s for s in slots if s["status"] == "empty"]
    assert len(empty) > 0