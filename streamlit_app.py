import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from app.calibration import list_calibrated_cameras, load_calibration
from app.gap_detector import DEFAULT_GAP_THRESHOLD_M, detect_gaps
from app.tracker import Sort
from app.vehicle_detector import VehicleDetector
from app.visualization import draw_spot_strip, draw_vehicles

st.set_page_config(page_title="Smart Parking Dashboard", layout="wide")
st.title("Smart Parking Dashboard")


@st.cache_resource
def get_detector(model_path, device):
    return VehicleDetector(model_path=model_path, device=device)


def render_spot_table(spots):
    st.dataframe(
        [
            {
                "ID": spot["id"],
                "Status": "Empty" if spot["status"] == "empty" else "Occupied",
                "Start (meters)": spot["start_m"],
                "End (meters)": spot["end_m"],
                "Length (meters)": spot["length_m"],
            }
            for spot in spots
        ],
        use_container_width=True,
    )


def render_metrics(vehicle_count, spots):
    occupied = sum(1 for spot in spots if spot["status"] == "occupied")
    empty = sum(1 for spot in spots if spot["status"] == "empty")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vehicles Detected", vehicle_count)
    col2.metric("Empty Spots", empty)
    col3.metric("Occupied Spots", occupied)


with st.sidebar:
    st.header("Settings")
    cameras = list_calibrated_cameras()
    if not cameras:
        st.warning("No cameras calibrated. Please use /calibrate first.")
    camera_id = st.selectbox("Camera", cameras) if cameras else None
    gap_threshold = st.slider("Empty spot threshold (meters)", 2.0, 8.0, DEFAULT_GAP_THRESHOLD_M, 0.5)
    conf = st.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
    device = st.selectbox("Device", ["cpu", "0"])
    model_path = st.text_input("Model weights path", "yolov8n.pt")

mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

if camera_id is None:
    st.stop()

matrix = load_calibration(camera_id)

if mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector = get_detector(model_path, device)
        vehicles = detector.detect(frame, conf=conf)
        spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold)

        annotated = draw_vehicles(frame, vehicles)
        strip = draw_spot_strip(spots)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Image with vehicle detections",
            )
            st.image(
                cv2.cvtColor(strip, cv2.COLOR_BGR2RGB),
                caption="Parking spot layout (Red=Occupied, Green=Empty)",
            )
        with col2:
            render_metrics(len(vehicles), spots)
            render_spot_table(spots)

else:
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    play_full_video = st.checkbox("Process full video (auto-play)", value=False)
    if not play_full_video:
        st.caption("In silent mode, only the first frame is processed as preview.")

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        temp_file.flush()

        detector = get_detector(model_path, device)
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        capture = cv2.VideoCapture(temp_file.name)
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0

        frame_placeholder = st.empty()
        strip_placeholder = st.empty()
        metrics_placeholder = st.empty()
        table_placeholder = st.empty()

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame, conf=conf)
            det_array = (
                np.array([[*d["bbox"], d["confidence"]] for d in detections])
                if detections
                else np.empty((0, 5))
            )
            tracked = tracker.update(det_array)
            vehicles = [
                {"bbox": [float(x1), float(y1), float(x2), float(y2)], "track_id": int(track_id)}
                for x1, y1, x2, y2, track_id in tracked
            ]
            spots = detect_gaps(matrix, vehicles, threshold_m=gap_threshold)

            annotated = draw_vehicles(frame, vehicles)
            strip = draw_spot_strip(spots)

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_index}",
                use_container_width=True,
            )
            strip_placeholder.image(
                cv2.cvtColor(strip, cv2.COLOR_BGR2RGB), use_container_width=True
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

st.caption("Data is processed in real-time and not stored.")