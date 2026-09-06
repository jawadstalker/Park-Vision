import cv2
import numpy as np


def draw_vehicles(frame_bgr, vehicles):
    annotated = frame_bgr.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = [int(c) for c in vehicle["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"id {vehicle['track_id']}" if "track_id" in vehicle else "car"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def draw_spot_strip(spots, width=800, height=80, margin=20):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (150, 150, 150), 1)

    if not spots:
        return canvas

    min_m = min(spot["start_m"] for spot in spots)
    max_m = max(spot["end_m"] for spot in spots)
    span = max(max_m - min_m, 1e-6)
    usable_width = width - 2 * margin

    for spot in spots:
        x1 = margin + int((spot["start_m"] - min_m) / span * usable_width)
        x2 = margin + int((spot["end_m"] - min_m) / span * usable_width)
        x2 = max(x2, x1 + 2)
        color = (60, 180, 75) if spot["status"] == "empty" else (60, 60, 220)
        cv2.rectangle(canvas, (x1, 15), (x2, height - 15), color, -1)
        cv2.rectangle(canvas, (x1, 15), (x2, height - 15), (30, 30, 30), 1)

    return canvas
