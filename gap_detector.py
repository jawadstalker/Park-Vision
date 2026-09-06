from typing import Dict, List, Optional

import numpy as np

from .perspective import bbox_bottom_edge, pixel_to_world

DEFAULT_GAP_THRESHOLD_M = 4.5


def vehicle_world_extent(matrix: np.ndarray, bbox: List[float]) -> tuple:
    left_px, right_px = bbox_bottom_edge(tuple(bbox))
    left_world = pixel_to_world(matrix, left_px)
    right_world = pixel_to_world(matrix, right_px)
    start = min(left_world[0], right_world[0])
    end = max(left_world[0], right_world[0])
    return start, end


def detect_gaps(
    matrix: np.ndarray,
    vehicles: List[Dict],
    threshold_m: float = DEFAULT_GAP_THRESHOLD_M,
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
            spots.append(
                {
                    "status": "empty",
                    "start_m": round(gap_start, 2),
                    "end_m": round(gap_end, 2),
                    "length_m": round(gap_length, 2),
                    "vehicle": None,
                }
            )

    spots.sort(key=lambda s: s["start_m"])
    for idx, spot in enumerate(spots, start=1):
        spot["id"] = idx

    return spots
