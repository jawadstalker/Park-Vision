import cv2
import numpy as np


EMPTY_COLOR = (60, 190, 80)
OCCUPIED_COLOR = (60, 70, 220)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)


def _zone_color(status: str):
    return EMPTY_COLOR if status == "empty" else OCCUPIED_COLOR


def draw_vehicles(frame_bgr, vehicles):
    annotated = frame_bgr.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = [int(c) for c in vehicle["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        track_id = vehicle.get("track_id")
        if track_id is not None:
            label = f"ID {track_id}"
        else:
            label = f"{vehicle.get('confidence', 0.0):.2f}"
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


def draw_parking_zones(frame_bgr, spots, alpha=0.28):
    """Draw configured parking polygons directly on the original frame."""
    annotated = frame_bgr.copy()
    overlay = frame_bgr.copy()

    for spot in spots:
        points = np.array(
            [[int(round(p["x"])), int(round(p["y"]))] for p in spot["points"]],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue

        color = _zone_color(spot["status"])
        cv2.fillPoly(overlay, [points], color)

    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    for spot in spots:
        points = np.array(
            [[int(round(p["x"])), int(round(p["y"]))] for p in spot["points"]],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue

        color = _zone_color(spot["status"])
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)

        moments = cv2.moments(points)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())

        status_label = "EMPTY" if spot["status"] == "empty" else "OCCUPIED"
        label = f"{spot['id']}  {status_label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tx = max(2, cx - tw // 2)
        ty = max(th + 2, cy + th // 2)
        cv2.rectangle(annotated, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), DARK, -1)
        cv2.putText(
            annotated,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_spot_strip(spots, width=800, height=80, margin=20):
    """Compatibility view: compact status strip for dashboards."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if not spots:
        return canvas

    ordered = sorted(spots, key=lambda spot: (spot.get("row", ""), str(spot["id"])))
    usable_width = width - 2 * margin
    slot_width = max(2, usable_width // max(len(ordered), 1))

    for index, spot in enumerate(ordered):
        x1 = margin + index * slot_width
        x2 = margin + (index + 1) * slot_width - 4
        color = _zone_color(spot["status"])
        cv2.rectangle(canvas, (x1, 15), (x2, height - 15), color, -1)
        cv2.rectangle(canvas, (x1, 15), (x2, height - 15), DARK, 1)
        cv2.putText(
            canvas,
            str(spot["id"]),
            (x1 + 6, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return canvas
