import numpy as np

from app.visualization import draw_spot_strip, draw_vehicles


def test_draw_vehicles_returns_same_shape_and_does_not_mutate_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    vehicles = [{"bbox": [10, 10, 50, 50], "track_id": 1}]

    result = draw_vehicles(frame, vehicles)

    assert result.shape == frame.shape
    assert not np.array_equal(result, original)
    assert np.array_equal(frame, original)


def test_draw_vehicles_empty_list_returns_unchanged_copy():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = draw_vehicles(frame, [])
    assert np.array_equal(result, frame)


def test_draw_spot_strip_empty_list_does_not_crash():
    strip = draw_spot_strip([])
    assert strip.shape == (80, 800, 3)


def test_draw_spot_strip_colors_empty_and_occupied_differently():
    spots = [
        {"status": "occupied", "start_m": 0, "end_m": 5},
        {"status": "empty", "start_m": 5, "end_m": 10},
    ]
    strip = draw_spot_strip(spots)

    red_mask = (strip[:, :, 2] > 150) & (strip[:, :, 0] < 100)
    green_mask = (strip[:, :, 1] > 150) & (strip[:, :, 2] < 100)

    assert red_mask.sum() > 0
    assert green_mask.sum() > 0
