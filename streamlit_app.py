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
st.title("داشبورد پارکینگ هوشمند")


@st.cache_resource
def get_detector(model_path, device):
    return VehicleDetector(model_path=model_path, device=device)


def render_spot_table(spots):
    st.dataframe(
        [
            {
                "شناسه": spot["id"],
                "وضعیت": "خالی" if spot["status"] == "empty" else "اشغال",
                "شروع (متر)": spot["start_m"],
                "پایان (متر)": spot["end_m"],
                "طول (متر)": spot["length_m"],
            }
            for spot in spots
        ],
        use_container_width=True,
    )


def render_metrics(vehicle_count, spots):
    occupied = sum(1 for spot in spots if spot["status"] == "occupied")
    empty = sum(1 for spot in spots if spot["status"] == "empty")
    col1, col2, col3 = st.columns(3)
    col1.metric("خودروهای شناسایی‌شده", vehicle_count)
    col2.metric("جای‌های خالی", empty)
    col3.metric("جای‌های اشغال", occupied)


with st.sidebar:
    st.header("تنظیمات")
    cameras = list_calibrated_cameras()
    if not cameras:
        st.warning("هیچ دوربینی کالیبره نشده. ابتدا از /calibrate استفاده کنید.")
    camera_id = st.selectbox("دوربین", cameras) if cameras else None
    gap_threshold = st.slider("آستانه جای خالی (متر)", 2.0, 8.0, DEFAULT_GAP_THRESHOLD_M, 0.5)
    conf = st.slider("آستانه اطمینان تشخیص", 0.1, 0.9, 0.35, 0.05)
    device = st.selectbox("device", ["cpu", "0"])
    model_path = st.text_input("مسیر وزن مدل", "yolov8n.pt")

mode = st.radio("نوع ورودی", ["تصویر", "ویدیو"], horizontal=True)

if camera_id is None:
    st.stop()

matrix = load_calibration(camera_id)

if mode == "تصویر":
    uploaded = st.file_uploader("آپلود تصویر", type=["jpg", "jpeg", "png"])
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
                caption="تصویر با تشخیص خودرو",
            )
            st.image(
                cv2.cvtColor(strip, cv2.COLOR_BGR2RGB),
                caption="نمای طرحی جای‌های پارک (قرمز=اشغال، سبز=خالی)",
            )
        with col2:
            render_metrics(len(vehicles), spots)
            render_spot_table(spots)

else:
    uploaded_video = st.file_uploader("آپلود ویدیو", type=["mp4", "avi", "mov"])
    play_full_video = st.checkbox("پردازش کامل ویدیو (پخش خودکار)", value=False)
    if not play_full_video:
        st.caption("در حالت خاموش، فقط فریم اول به‌عنوان پیش‌نمایش پردازش می‌شود.")

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
                caption=f"فریم {frame_index}",
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

st.caption("داده‌ها لحظه‌ای پردازش می‌شوند و ذخیره نمی‌گردند.")
