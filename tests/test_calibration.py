import numpy as np

from app.calibration import compute_homography, list_calibrated_cameras, load_calibration, save_calibration


def test_compute_homography_shape():
    pixel_points = [(0, 480), (640, 480), (640, 300), (0, 300)]
    real_world_points = [(0, 0), (20, 0), (20, 6), (0, 6)]
    matrix = compute_homography(pixel_points, real_world_points)
    assert matrix.shape == (3, 3)


def test_save_and_load_roundtrip():
    matrix = np.eye(3)
    save_calibration("cam-a", matrix)
    loaded = load_calibration("cam-a")
    assert loaded is not None
    assert np.allclose(loaded, matrix)


def test_load_missing_camera_returns_none():
    assert load_calibration("does-not-exist") is None


def test_list_calibrated_cameras_sorted():
    save_calibration("cam-b", np.eye(3))
    save_calibration("cam-a", np.eye(3))
    assert list_calibrated_cameras() == ["cam-a", "cam-b"]


def test_list_calibrated_cameras_empty_when_none_saved():
    assert list_calibrated_cameras() == []
