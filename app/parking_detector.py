from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def vehicle_ground_point(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_zone(point: Tuple[float, float], polygon: List[Dict]) -> bool:
    contour = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.float32)
    if len(contour) < 3:
        return False
    return cv2.pointPolygonTest(contour, point, False) >= 0


def polygon_area(points: List[Dict]) -> float:
    contour = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    if len(contour) < 3:
        return 0.0
    return float(abs(cv2.contourArea(contour)))


def zone_center(points: List[Dict]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return (
        sum(float(p["x"]) for p in points) / len(points),
        sum(float(p["y"]) for p in points) / len(points),
    )


def detect_parking_zones(
    vehicles: List[Dict],
    zones: List[Dict],
    min_confidence: float = 0.0,
) -> List[Dict]:
    """Assign each vehicle to at most one fixed parking zone.

    Occupancy is based on the bottom-center point of a vehicle bounding box.
    This point approximates the vehicle's contact point with the ground and is
    more stable under perspective than requiring the whole bbox to fit inside
    the parking polygon.
    """
    usable_vehicles = [
        vehicle for vehicle in vehicles
        if float(vehicle.get("confidence", 1.0)) >= min_confidence
    ]

    assignments: Dict[str, Optional[Dict]] = {str(zone["id"]): None for zone in zones}
    candidates = []

    for vehicle_index, vehicle in enumerate(usable_vehicles):
        point = vehicle_ground_point(vehicle["bbox"])
        for zone in zones:
            if point_in_zone(point, zone["points"]):
                center = zone_center(zone["points"])
                distance = (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
                candidates.append((distance, vehicle_index, zone["id"], point))

    # Resolve the rare case where multiple detections fall in one zone.
    # Prefer the detection whose ground point is closest to the zone center.
    candidates.sort(key=lambda item: item[0])
    used_vehicles = set()
    for _, vehicle_index, zone_id, point in candidates:
        zone_key = str(zone_id)
        if assignments[zone_key] is not None or vehicle_index in used_vehicles:
            continue
        assignments[zone_key] = usable_vehicles[vehicle_index]
        assignments[zone_key]["ground_point"] = {
            "x": float(point[0]),
            "y": float(point[1]),
        }
        used_vehicles.add(vehicle_index)

    results = []
    for zone in zones:
        zone_id = str(zone["id"])
        vehicle = assignments[zone_id]
        results.append(
            {
                "id": zone_id,
                "row": str(zone.get("row", "A")),
                "points": zone["points"],
                "status": "occupied" if vehicle is not None else "empty",
                "vehicle": vehicle,
                "vehicle_id": vehicle.get("track_id") if vehicle else None,
                "area_px": round(polygon_area(zone["points"]), 1),
            }
        )

    return results
