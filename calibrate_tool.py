import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from app.calibration import compute_homography, load_calibration, save_calibration

st.set_page_config(page_title="Camera Calibration Tool", layout="wide")
st.title("Camera Calibration Tool")
st.caption(
    "Click on 4 specific points in the image (e.g., table corners or street markings) and "
    "enter the actual real-world distance of each point on the ground (in meters)."
)

if "calibration_points" not in st.session_state:
    st.session_state.calibration_points = []

with st.sidebar:
    st.header("Settings")
    camera_id = st.text_input("Camera ID", "street-01")
    if st.button("Clear selected points"):
        st.session_state.calibration_points = []

    existing = load_calibration(camera_id)
    if existing is not None:
        st.info(f"Camera '{camera_id}' is already calibrated. Saving again will overwrite it.")

uploaded_image = st.file_uploader("Upload reference camera image", type=["jpg", "jpeg", "png"])

if uploaded_image is None:
    st.stop()

file_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
reference_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
reference_image_rgb = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2RGB)

col_image, col_points = st.columns([2, 1])

with col_image:
    st.subheader("Image (click on a point)")
    click = streamlit_image_coordinates(reference_image_rgb, key="calibration_click")

with col_points:
    st.subheader("New Point")
    if click is not None:
        st.write(f"Selected pixel: ({click['x']}, {click['y']})")
        real_x = st.number_input("Real-world X coordinate (meters)", value=0.0, step=0.5, key="real_x_input")
        real_y = st.number_input("Real-world Y coordinate (meters)", value=0.0, step=0.5, key="real_y_input")
        if st.button("Add this point"):
            if len(st.session_state.calibration_points) >= 4:
                st.warning("4 points are enough. To start over, clear the points first.")
            else:
                st.session_state.calibration_points.append(
                    {
                        "pixel": (float(click["x"]), float(click["y"])),
                        "real_world": (float(real_x), float(real_y)),
                    }
                )
                st.rerun()
    else:
        st.write("Click on the image to select a point.")

st.subheader(f"Selected points ({len(st.session_state.calibration_points)}/4)")
st.dataframe(
    [
        {
            "Index": i + 1,
            "Pixel X": p["pixel"][0],
            "Pixel Y": p["pixel"][1],
            "Real X (meters)": p["real_world"][0],
            "Real Y (meters)": p["real_world"][1],
        }
        for i, p in enumerate(st.session_state.calibration_points)
    ],
    use_container_width=True,
)

if len(st.session_state.calibration_points) == 4:
    pixel_points = [p["pixel"] for p in st.session_state.calibration_points]
    real_world_points = [p["real_world"] for p in st.session_state.calibration_points]
    matrix = compute_homography(pixel_points, real_world_points)

    warped_width, warped_height = 640, 480
    scale = 10
    warped = cv2.warpPerspective(
        reference_image_bgr,
        matrix,
        (warped_width, warped_height),
    )

    st.subheader("Top-view preview (to verify calibration accuracy)")
    st.image(
        cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
        caption="If street/table lines appear straight and parallel in this image, calibration is correct.",
        use_column_width=True,
        
    )

    if st.button("Save calibration", type="primary"):
        save_calibration(camera_id, matrix)
        st.success(f"Calibration for camera '{camera_id}' saved.")
else:
    st.info("4 points are required to compute calibration.")