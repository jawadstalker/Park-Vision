import json
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

CALIBRATION_DIR = os.environ.get("CALIBRATION_DIR", "calibrations")


def _ensure_dir() -> None:
    os.makedirs(CALIBRATION_DIR, exist_ok=True)


def _path_for(camera_id: str) -> str:
    return os.path.join(CALIBRATION_DIR, f"{camera_id}.json")


def compute_homography(
    pixel_points: List[Tuple[float, float]],
    real_world_points: List[Tuple[float, float]],
) -> np.ndarray:
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(real_world_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def save_calibration(camera_id: str, matrix: np.ndarray) -> None:
    _ensure_dir()
    with open(_path_for(camera_id), "w") as f:
        json.dump({"matrix": matrix.tolist(), "saved_at": time.time()}, f)


def load_calibration(camera_id: str) -> Optional[np.ndarray]:
    path = _path_for(camera_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return np.array(data["matrix"], dtype=np.float32)


def list_calibrated_cameras() -> List[str]:
    _ensure_dir()
    return sorted(f[:-5] for f in os.listdir(CALIBRATION_DIR) if f.endswith(".json"))
