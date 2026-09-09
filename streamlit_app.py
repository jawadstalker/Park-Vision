import json
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from app.parking_detector import detect_parking_zones
from app.parking_zones import list_zoned_cameras, load_zones, save_zones
from app.tracker import Sort
from app.vehicle_detector import VehicleDetector
from app.visualization import draw_parking_zones, draw_spot_strip, draw_vehicles


st.set_page_config(page_title="Smart Parking Dashboard", layout="wide")
st.title("Smart Parking Dashboard")
st.caption("Fixed parking zones · Two-row parking · Vehicle-based occupancy")


@st.cache_resource
def get_detector(model_path, device):
    return VehicleDetector(model_path=model_path, device=device)


def default_zone_template():
    return [
        {
            "id": "A1",
            "row": "A",
            "points": [
                {"x": 100, "y": 100},
                {"x": 200, "y": 100},
                {"x": 210, "y": 220},
                {"x": 90, "y": 220},
            ],
        }
    ]


def zones_to_text(zones):
    return json.dumps(zones, ensure_ascii=False, indent=2)


def parse_zones(text):
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError("Zones must be a non-empty JSON array.")

    normalized = []
    ids = set()
    for zone in raw:
        if not isinstance(zone, dict):
            raise ValueError("Every zone must be an object.")
        zone_id = str(zone.get("id", "")).strip()
        if not zone_id:
            raise ValueError("Every zone needs an id, e.g. A1 or B3.")
        if zone_id in ids:
            raise ValueError(f"Duplicate zone id: {zone_id}")
        ids.add(zone_id)

        points = zone.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Zone {zone_id} needs at least 3 points.")

        normalized_points = []
        for point in points:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(f"Zone {zone_id} has an invalid point.")
            normalized_points.append({"x": float(point["x"]), "y": float(point["y"])})

        normalized.append(
            {
                "id": zone_id,
                "row": str(zone.get("row", "A")),
                "points": normalized_points,
            }
        )

    return normalized


def render_metrics(vehicle_count, spots):
    occupied = sum(spot["status"] == "occupied" for spot in spots)
    empty = len(spots) - occupied
    total = len(spots)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vehicles Detected", vehicle_count)
    col2.metric("Total Spots", total)
    col3.metric("Empty Spots", empty)
    col4.metric("Occupied Spots", occupied)


def render_spot_table(spots):
    rows = []
    for spot in spots:
        rows.append(
            {
                "ID": spot["id"],
                "Row": spot.get("row", ""),
                "Status": "Empty" if spot["status"] == "empty" else "Occupied",
                "Vehicle": spot.get("vehicle_id") or "—",
            }
        )
    st.dataframe(rows, hide_index=True)


def process_frame(frame, detector, conf, zones, tracker=None):
    detections = detector.detect(frame, conf=conf)

    if tracker is not None:
        det_array = (
            np.array([[*d["bbox"], d["confidence"]] for d in detections], dtype=np.float32)
            if detections
            else np.empty((0, 5), dtype=np.float32)
        )
        tracked = tracker.update(det_array)
        vehicles = [
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "track_id": int(track_id),
                "confidence": 1.0,
            }
            for x1, y1, x2, y2, track_id in tracked
        ]
    else:
        vehicles = detections

    spots = detect_parking_zones(vehicles, zones, min_confidence=0.0)
    annotated = draw_parking_zones(frame, spots)
    annotated = draw_vehicles(annotated, vehicles)
    return annotated, vehicles, spots


with st.sidebar:
    st.header("Settings")

    zoned_cameras = list_zoned_cameras()
    camera_options = zoned_cameras if zoned_cameras else ["street-01"]
    camera_id = st.selectbox("Camera", camera_options)

    conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    device = st.selectbox("Device", ["cpu", "0"], index=0)
    model_path = st.text_input("Model weights", "yolov8n.pt")

    st.divider()
    st.subheader("Parking zones")
    zone_text_default = zones_to_text(load_zones(camera_id) or default_zone_template())
    zone_text = st.text_area(
        "Zone configuration JSON",
        value=zone_text_default,
        height=330,
        help=(
            "Configure once per camera. Each zone is a polygon in image pixels. "
            "Use row A/B for the two opposing parking rows."
        ),
    )

    if st.button("Save zones", type="primary", use_container_width=True):
        try:
            zones = parse_zones(zone_text)
            save_zones(camera_id, zones)
            st.success(f"Saved {len(zones)} zones for {camera_id}.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    existing_zones = load_zones(camera_id)
    if existing_zones:
        st.caption(f"Loaded {len(existing_zones)} saved zones for {camera_id}.")
    else:
        st.warning("No saved zones yet. Save the polygons before detection.")

zones = load_zones(camera_id)
if not zones:
    st.info("Configure and save the parking zones in the sidebar first.")
    st.stop()

mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

detector = get_detector(model_path, device)

if mode == "Image":
    uploaded = st.file_uploader("Upload parking image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode the image.")
            st.stop()

        annotated, vehicles, spots = process_frame(frame, detector, conf, zones)

        col1, col2 = st.columns([2.2, 1])
        with col1:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Parking zones: green = empty · red = occupied",
            )
            st.image(
                cv2.cvtColor(draw_spot_strip(spots), cv2.COLOR_BGR2RGB),
                caption="Zone status overview",
            )
        with col2:
            render_metrics(len(vehicles), spots)
            render_spot_table(spots)

else:
    uploaded_video = st.file_uploader("Upload parking video", type=["mp4", "avi", "mov"])
    play_full_video = st.checkbox("Process full video", value=False)

    if uploaded_video is not None:
        suffix = ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_video.read())
        temp_file.flush()
        temp_file.close()

        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        capture = cv2.VideoCapture(temp_file.name)
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        table_placeholder = st.empty()

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, vehicles, spots = process_frame(
                frame,
                detector,
                conf,
                zones,
                tracker=tracker,
            )

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_index} · green=empty · red=occupied",
            )

            with metrics_placeholder.container():
                render_metrics(len(vehicles), spots)
            with table_placeholder.container():
                render_spot_table(spots)

            frame_index += 1
            if not play_full_video:
                break
            time.sleep(max(0.0, 1.0 / fps))

        capture.release()

st.caption("Parking zones are configured once per camera and reused for every image/video frame.")
