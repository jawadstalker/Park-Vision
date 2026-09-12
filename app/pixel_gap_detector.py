from typing import Dict, List

import numpy as np

from .tracker import iou_batch

DEFAULT_MIN_GAP_RATIO = 0.85
DEFAULT_ROW_TOLERANCE_RATIO = 0.6
DEFAULT_RECHECK_CONF = 0.15
DEFAULT_GAP_TRIGGER_RATIO = 1.5
DEFAULT_DEDUPE_IOU = 0.3


def _subdivide_edge_gap(gap_start: float, gap_end: float, unit_width: float, edge_bbox: List[float]) -> List[Dict]:
    gap_width = gap_end - gap_start
    spot_count = max(1, int(gap_width // unit_width))
    used = unit_width * spot_count
    margin = (gap_width - used) / 2.0

    spots = []
    for k in range(spot_count):
        x1 = gap_start + margin + k * unit_width
        x2 = x1 + unit_width
        spots.append({"status": "empty", "bbox": [x1, edge_bbox[1], x2, edge_bbox[3]]})
    return spots


def detect_pixel_gaps(
    vehicles: List[Dict],
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    frame_width: float = None,
) -> List[Dict]:
    if not vehicles:
        return []

    sorted_vehicles = sorted(vehicles, key=lambda v: (v["bbox"][0] + v["bbox"][2]) / 2.0)

    slots: List[Dict] = []
    for vehicle in sorted_vehicles:
        slots.append({"status": "occupied", "bbox": list(vehicle["bbox"])})

    for i in range(len(sorted_vehicles) - 1):
        left = sorted_vehicles[i]["bbox"]
        right = sorted_vehicles[i + 1]["bbox"]

        left_width = left[2] - left[0]
        right_width = right[2] - right[0]
        local_unit_width = (left_width + right_width) / 2.0

        gap_start = left[2]
        gap_end = right[0]
        gap_width = gap_end - gap_start

        if gap_width >= local_unit_width * min_gap_ratio:
            spot_count = max(1, int(gap_width // local_unit_width))
            used = local_unit_width * spot_count
            margin = (gap_width - used) / 2.0

            for k in range(spot_count):
                x1 = gap_start + margin + k * local_unit_width
                x2 = x1 + local_unit_width
                center_x = (x1 + x2) / 2.0
                frac = (center_x - gap_start) / max(gap_width, 1e-6)
                top = left[1] + (right[1] - left[1]) * frac
                bottom = left[3] + (right[3] - left[3]) * frac
                slots.append({"status": "empty", "bbox": [x1, top, x2, bottom]})

    if frame_width is not None:
        first = sorted_vehicles[0]["bbox"]
        first_width = first[2] - first[0]
        left_edge_width = first[0] - 0.0
        if left_edge_width >= first_width * min_gap_ratio:
            slots.extend(_subdivide_edge_gap(0.0, first[0], first_width, first))

        last = sorted_vehicles[-1]["bbox"]
        last_width = last[2] - last[0]
        right_edge_width = frame_width - last[2]
        if right_edge_width >= last_width * min_gap_ratio:
            slots.extend(_subdivide_edge_gap(last[2], frame_width, last_width, last))

    slots.sort(key=lambda s: s["bbox"][0])
    for idx, slot in enumerate(slots, start=1):
        slot["id"] = idx

    return slots


def cluster_vehicles_by_row(
    vehicles: List[Dict], row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO
) -> List[List[Dict]]:
    if not vehicles:
        return []

    def y_center(v):
        return (v["bbox"][1] + v["bbox"][3]) / 2.0

    def height(v):
        return v["bbox"][3] - v["bbox"][1]

    sorted_by_y = sorted(vehicles, key=y_center)
    avg_height = sum(height(v) for v in vehicles) / len(vehicles)
    tolerance = avg_height * row_tolerance_ratio

    rows: List[List[Dict]] = []
    current_row = [sorted_by_y[0]]
    current_y = y_center(sorted_by_y[0])

    for vehicle in sorted_by_y[1:]:
        if abs(y_center(vehicle) - current_y) <= tolerance:
            current_row.append(vehicle)
            current_y = sum(y_center(v) for v in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [vehicle]
            current_y = y_center(vehicle)

    rows.append(current_row)
    return rows


def detect_pixel_gaps_multi_row(
    vehicles: List[Dict],
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    frame_width: float = None,
) -> List[Dict]:
    rows = cluster_vehicles_by_row(vehicles, row_tolerance_ratio)

    all_slots: List[Dict] = []
    for row in rows:
        all_slots.extend(detect_pixel_gaps(row, min_gap_ratio=min_gap_ratio, frame_width=frame_width))

    all_slots.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    for idx, slot in enumerate(all_slots, start=1):
        slot["id"] = idx

    return all_slots


def refine_with_low_confidence_recheck(
    detector,
    image_bgr: np.ndarray,
    vehicles: List[Dict],
    recheck_conf: float = DEFAULT_RECHECK_CONF,
    gap_trigger_ratio: float = DEFAULT_GAP_TRIGGER_RATIO,
    dedupe_iou: float = DEFAULT_DEDUPE_IOU,
) -> List[Dict]:
    if len(vehicles) < 2:
        return vehicles

    sorted_vehicles = sorted(vehicles, key=lambda v: (v["bbox"][0] + v["bbox"][2]) / 2.0)
    augmented = list(vehicles)

    for i in range(len(sorted_vehicles) - 1):
        left = sorted_vehicles[i]["bbox"]
        right = sorted_vehicles[i + 1]["bbox"]

        local_unit_width = ((left[2] - left[0]) + (right[2] - right[0])) / 2.0
        gap_start, gap_end = left[2], right[0]
        gap_width = gap_end - gap_start

        if gap_width < local_unit_width * gap_trigger_ratio:
            continue

        y1 = int(min(left[1], right[1]))
        y2 = int(max(left[3], right[3]))
        x1 = int(gap_start)
        x2 = int(gap_end)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        extra_detections = detector.detect(crop, conf=recheck_conf)
        for detection in extra_detections:
            ex1, ey1, ex2, ey2 = detection["bbox"]
            full_bbox = [ex1 + x1, ey1 + y1, ex2 + x1, ey2 + y1]

            existing_boxes = np.array([v["bbox"] for v in augmented])
            candidate_box = np.array([full_bbox])
            if len(existing_boxes) and iou_batch(candidate_box, existing_boxes).max() > dedupe_iou:
                continue

            augmented.append({"bbox": full_bbox, "confidence": detection["confidence"]})

    return augmented