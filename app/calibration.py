import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .database import list_calibration_camera_ids, load_calibration_record, save_calibration_record


class BadCalibrationError(ValueError):
    """Raised when the supplied points imply an unusable camera angle/geometry."""


MAX_SCALE_RATIO = 8.0


def _edge_scale_m_per_px(pixel_points, real_world_points, i: int, j: int) -> Optional[float]:
    px = np.array(pixel_points[i]) - np.array(pixel_points[j])
    wd = np.array(real_world_points[i]) - np.array(real_world_points[j])
    px_dist = float(np.linalg.norm(px))
    wd_dist = float(np.linalg.norm(wd))
    if px_dist < 1e-6:
        return None
    return wd_dist / px_dist


def _check_scale_consistency(pixel_points, real_world_points) -> None:
    """Compares meters-per-pixel across opposite edges of the calibration quad.

    A very oblique/low camera angle (the "bad angle" cases from real footage)
    makes near objects span far more pixels per meter than far objects, so a
    single homography can't represent distances reliably across the frame.
    Points 0-1 and 3-2 are treated as opposite edges (matching the point
    order used by calibrate_tool.py / the /calibrate payload: near-left,
    near-right, far-right, far-left).
    """
    if len(pixel_points) != 4:
        return
    scale_near = _edge_scale_m_per_px(pixel_points, real_world_points, 0, 1)
    scale_far = _edge_scale_m_per_px(pixel_points, real_world_points, 3, 2)
    if not scale_near or not scale_far:
        return
    ratio = max(scale_near, scale_far) / min(scale_near, scale_far)
    if ratio > MAX_SCALE_RATIO:
        raise BadCalibrationError(
            "Calibration points imply a meters-per-pixel scale that varies "
            f"{ratio:.1f}x between the near and far edges. This usually means "
            "the camera is too low/oblique for reliable distance measurement "
            "(see camera-angle guidance) or the points were picked incorrectly."
        )


def compute_homography(
    pixel_points: List[Tuple[float, float]],
    real_world_points: List[Tuple[float, float]],
) -> np.ndarray:
    _check_scale_consistency(pixel_points, real_world_points)

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(real_world_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)

    if not np.all(np.isfinite(matrix)):
        raise BadCalibrationError("Computed homography contains non-finite values.")
    if abs(np.linalg.det(matrix)) < 1e-9:
        raise BadCalibrationError(
            "Calibration points are (near-)collinear or degenerate; pick four "
            "points that form a proper quadrilateral on the ground plane."
        )

    return matrix


def save_calibration(camera_id: str, matrix: np.ndarray) -> None:
    save_calibration_record(camera_id, json.dumps(matrix.tolist()))


def load_calibration(camera_id: str) -> Optional[np.ndarray]:
    matrix_json = load_calibration_record(camera_id)
    if matrix_json is None:
        return None
    return np.array(json.loads(matrix_json), dtype=np.float32)


def list_calibrated_cameras() -> List[str]:
    return list_calibration_camera_ids()
