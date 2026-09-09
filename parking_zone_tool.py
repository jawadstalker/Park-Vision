import json

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from app.parking_zones import load_zones, save_zones


st.set_page_config(page_title="Parking Zone Calibrator", layout="wide")
st.title("Parking Zone Configuration")
st.caption("Configure fixed parking polygons once per camera. They are reused for every frame.")


if "zone_points" not in st.session_state:
    st.session_state.zone_points = []
if "zones" not in st.session_state:
    st.session_state.zones = []
if "source_key" not in st.session_state:
    st.session_state.source_key = None


with st.sidebar:
    st.header("Camera")
    camera_id = st.text_input("Camera ID", "street-01")

    st.divider()
    st.header("Current zone")
    zone_id = st.text_input("Zone ID", f"A{len(st.session_state.zones) + 1}")
    row = st.selectbox("Row", ["A", "B"])

    col1, col2 = st.columns(2)
    if col1.button("Undo point") and st.session_state.zone_points:
        st.session_state.zone_points.pop()
        st.rerun()
    if col2.button("Clear points"):
        st.session_state.zone_points = []
        st.rerun()

    if st.button("Add zone", type="primary", use_container_width=True):
        if len(st.session_state.zone_points) < 3:
            st.error("A parking zone needs at least 3 points.")
        elif not zone_id.strip():
            st.error("Zone ID is required.")
        elif any(zone["id"] == zone_id.strip() for zone in st.session_state.zones):
            st.error(f"Zone {zone_id} already exists.")
        else:
            st.session_state.zones.append(
                {
                    "id": zone_id.strip(),
                    "row": row,
                    "points": [
                        {"x": float(x), "y": float(y)}
                        for x, y in st.session_state.zone_points
                    ],
                }
            )
            st.session_state.zone_points = []
            st.rerun()

    st.divider()
    st.subheader("Configured zones")
    for index, zone in enumerate(st.session_state.zones):
        st.write(f"**{zone['id']}** · Row {zone['row']} · {len(zone['points'])} points")
        if st.button(f"Delete {zone['id']}", key=f"delete_{index}"):
            st.session_state.zones.pop(index)
            st.rerun()

    if st.button("Save all zones", type="primary", use_container_width=True):
        if not st.session_state.zones:
            st.error("Add at least one parking zone first.")
        else:
            save_zones(camera_id, st.session_state.zones)
            st.success(f"Saved {len(st.session_state.zones)} zones for {camera_id}.")

    if st.button("Load saved zones", use_container_width=True):
        saved = load_zones(camera_id)
        if saved:
            st.session_state.zones = saved
            st.session_state.zone_points = []
            st.success(f"Loaded {len(saved)} zones.")
            st.rerun()
        else:
            st.warning("No saved zones for this camera.")

uploaded = st.file_uploader(
    "Upload one representative image from this camera",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Upload a representative camera image, then click the corners of each parking space.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if frame is None:
    st.error("Could not decode the image.")
    st.stop()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Keep the widget key stable for this uploaded image so click events work
# without losing the current polygon across Streamlit reruns.
source_key = f"{camera_id}_{frame.shape[1]}x{frame.shape[0]}_{uploaded.name}"
if st.session_state.source_key != source_key:
    st.session_state.source_key = source_key
    st.session_state.zone_points = []

st.subheader("Click polygon corners")
st.write(
    "Click 4 corners (or more) around one parking space. "
    "The clicks are stored as image pixel coordinates, not world coordinates."
)

value = streamlit_image_coordinates(
    rgb,
    key=f"parking_image_{source_key}",
    cursor="crosshair",
)

if value is not None:
    click_key = (int(value["x"]), int(value["y"]), int(value.get("timestamp", 0)))
    last_click = st.session_state.get("last_click")
    if last_click != click_key:
        st.session_state.last_click = click_key
        st.session_state.zone_points.append((click_key[0], click_key[1]))
        st.rerun()

st.subheader("Current polygon")
if st.session_state.zone_points:
    preview = frame.copy()
    pts = np.array(st.session_state.zone_points, dtype=np.int32)
    for index, (x, y) in enumerate(st.session_state.zone_points, start=1):
        cv2.circle(preview, (x, y), 5, (255, 180, 0), -1)
        cv2.putText(
            preview,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )
    if len(pts) >= 2:
        cv2.polylines(preview, [pts], False, (255, 180, 0), 2)
    if len(pts) >= 3:
        cv2.polylines(preview, [pts], True, (255, 180, 0), 2)

    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Current parking-zone polygon")
    st.code(json.dumps(st.session_state.zone_points, indent=2))
else:
    st.caption("No points selected yet.")

if st.session_state.zones:
    st.subheader("Saved in this session")
    st.json(st.session_state.zones)
