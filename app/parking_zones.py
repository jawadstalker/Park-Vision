import json
import os
from typing import Dict, List, Optional

ZONES_DIR = os.environ.get("PARKING_ZONES_DIR", "parking_zones")


def _ensure_dir() -> None:
    os.makedirs(ZONES_DIR, exist_ok=True)


def _path_for(camera_id: str) -> str:
    safe_camera_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_id)
    return os.path.join(ZONES_DIR, f"{safe_camera_id}.json")


def normalize_zone(zone: Dict) -> Dict:
    points = zone.get("points", [])
    normalized = []
    for point in points:
        if isinstance(point, dict):
            normalized.append({"x": float(point["x"]), "y": float(point["y"])})
        else:
            normalized.append({"x": float(point[0]), "y": float(point[1])})

    return {
        "id": str(zone["id"]),
        "row": str(zone.get("row", "A")),
        "points": normalized,
    }


def save_zones(camera_id: str, zones: List[Dict]) -> None:
    _ensure_dir()
    payload = {"camera_id": camera_id, "zones": [normalize_zone(z) for z in zones]}
    with open(_path_for(camera_id), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_zones(camera_id: str) -> Optional[List[Dict]]:
    path = _path_for(camera_id)
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    return [normalize_zone(zone) for zone in payload.get("zones", [])]


def list_zoned_cameras() -> List[str]:
    _ensure_dir()
    return sorted(
        filename[:-5]
        for filename in os.listdir(ZONES_DIR)
        if filename.endswith(".json")
    )
