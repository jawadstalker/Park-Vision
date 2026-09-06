from typing import Tuple

import cv2
import numpy as np


def pixel_to_world(matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    src = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)
    return float(dst[0][0][0]), float(dst[0][0][1])


def bbox_bottom_edge(bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    return (x1, y2), (x2, y2)
