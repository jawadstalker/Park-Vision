import time

import cv2
import numpy as np

from .gap_detector import DEFAULT_GAP_THRESHOLD_M, detect_gaps
from .tracker import Sort


def process_video(video_path, detector, matrix, gap_threshold_m=DEFAULT_GAP_THRESHOLD_M, conf=0.35):
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detector.detect(frame, conf=conf)
        if detections:
            det_array = np.array(
                [[*d["bbox"], d["confidence"]] for d in detections], dtype=float
            )
        else:
            det_array = np.empty((0, 5))

        start_time = time.perf_counter()
        tracked = tracker.update(det_array)
        vehicles = [
            {"bbox": [float(x1), float(y1), float(x2), float(y2)], "track_id": int(track_id)}
            for x1, y1, x2, y2, track_id in tracked
        ]
        spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold_m)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        yield {
            "frame_index": frame_index,
            "timestamp_s": round(frame_index / fps, 3),
            "vehicles": vehicles,
            "spots": spots,
            "tracking_gap_time_ms": round(elapsed_ms, 2),
        }

        frame_index += 1

    capture.release()
