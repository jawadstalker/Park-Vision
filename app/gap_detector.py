from typing import Dict, List, Optional

import numpy as np

from .perspective import bbox_bottom_edge, pixel_to_world

DEFAULT_GAP_THRESHOLD_M = 4.5
DEFAULT_SPOT_LENGTH_M = 5.0


def vehicle_world_extent(matrix: np.ndarray, bbox: List[float]) -> tuple:
    left_px, right_px = bbox_bottom_edge(tuple(bbox))
    left_world = pixel_to_world(matrix, left_px)
    right_world = pixel_to_world(matrix, right_px)
    start = min(left_world[0], right_world[0])
    end = max(left_world[0], right_world[0])
    return start, end


def split_gap_into_spots(gap_start: float, gap_end: float, spot_length_m: float) -> List[Dict]:
    gap_length = gap_end - gap_start
    spot_count = max(1, int(gap_length // spot_length_m))
    used_length = spot_length_m * spot_count

    # Distribute any leftover space evenly as margin on both sides, so spots
    # sit centered in the gap rather than flush against one edge.
    margin = (gap_length - used_length) / 2.0

    spots = []
    for i in range(spot_count):
        spot_start = gap_start + margin + i * spot_length_m
        spot_end = spot_start + spot_length_m
        spots.append(
            {
                "status": "empty",
                "start_m": round(spot_start, 2),
                "end_m": round(spot_end, 2),
                "length_m": round(spot_end - spot_start, 2),
                "vehicle": None,
            }
        )
    return spots


def detect_gaps(
    matrix: np.ndarray,
    vehicles: List[Dict],
    threshold_m: float = DEFAULT_GAP_THRESHOLD_M,
    spot_length_m: float = DEFAULT_SPOT_LENGTH_M,
) -> List[Dict]:
    extents = []
    for vehicle in vehicles:
        start, end = vehicle_world_extent(matrix, vehicle["bbox"])
        extents.append({"vehicle": vehicle, "start": start, "end": end})
    extents.sort(key=lambda e: e["start"])

    spots: List[Dict] = []

    for extent in extents:
        spots.append(
            {
                "status": "occupied",
                "start_m": round(extent["start"], 2),
                "end_m": round(extent["end"], 2),
                "length_m": round(extent["end"] - extent["start"], 2),
                "vehicle": extent["vehicle"],
            }
        )

    for i in range(len(extents) - 1):
        gap_start = extents[i]["end"]
        gap_end = extents[i + 1]["start"]
        gap_length = gap_end - gap_start
        if gap_length >= threshold_m:
            spots.extend(split_gap_into_spots(gap_start, gap_end, spot_length_m))

    spots.sort(key=lambda s: s["start_m"])
    for idx, spot in enumerate(spots, start=1):
        spot["id"] = idx

    return spots