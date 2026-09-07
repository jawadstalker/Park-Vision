import numpy as np

from app.calibration import compute_homography
from app.perspective import bbox_bottom_edge, pixel_to_world


def test_bbox_bottom_edge():
    left, right = bbox_bottom_edge((10, 20, 50, 80))
    assert left == (10, 80)
    assert right == (50, 80)


def test_pixel_to_world_identity_matrix():
    matrix = np.eye(3)
    x, y = pixel_to_world(matrix, (5, 7))
    assert abs(x - 5) < 1e-6
    assert abs(y - 7) < 1e-6


def test_pixel_to_world_with_real_homography():
    pixel_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    real_world_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    matrix = compute_homography(pixel_points, real_world_points)

    x1, y1 = pixel_to_world(matrix, (0, 100))
    assert abs(x1 - 0) < 1e-3
    assert abs(y1 - 0) < 1e-3

    x2, y2 = pixel_to_world(matrix, (100, 100))
    assert abs(x2 - 10) < 1e-3
    assert abs(y2 - 0) < 1e-3
