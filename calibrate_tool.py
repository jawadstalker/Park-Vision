import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from app.calibration import compute_homography, load_calibration, save_calibration

st.set_page_config(page_title="ابزار کالیبراسیون دوربین", layout="wide")
st.title("ابزار کالیبراسیون دوربین")
st.caption(
    "روی ۴ نقطه‌ی مشخص در تصویر (مثلاً گوشه‌های جدول یا خط‌کشی خیابان) کلیک کنید و "
    "فاصله واقعی هر نقطه را روی زمین (بر حسب متر) وارد کنید."
)

if "calibration_points" not in st.session_state:
    st.session_state.calibration_points = []

with st.sidebar:
    st.header("تنظیمات")
    camera_id = st.text_input("شناسه دوربین (camera_id)", "street-01")
    if st.button("پاک کردن نقاط انتخاب‌شده"):
        st.session_state.calibration_points = []

    existing = load_calibration(camera_id)
    if existing is not None:
        st.info(f"دوربین «{camera_id}» از قبل کالیبره شده. ذخیره مجدد آن را بازنویسی می‌کند.")

uploaded_image = st.file_uploader("آپلود تصویر رفرنس دوربین", type=["jpg", "jpeg", "png"])

if uploaded_image is None:
    st.stop()

file_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
reference_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
reference_image_rgb = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2RGB)

col_image, col_points = st.columns([2, 1])

with col_image:
    st.subheader("تصویر (روی یک نقطه کلیک کنید)")
    click = streamlit_image_coordinates(reference_image_rgb, key="calibration_click")

with col_points:
    st.subheader("نقطه جدید")
    if click is not None:
        st.write(f"پیکسل انتخاب‌شده: ({click['x']}, {click['y']})")
        real_x = st.number_input("مختصات واقعی X (متر)", value=0.0, step=0.5, key="real_x_input")
        real_y = st.number_input("مختصات واقعی Y (متر)", value=0.0, step=0.5, key="real_y_input")
        if st.button("افزودن این نقطه"):
            if len(st.session_state.calibration_points) >= 4:
                st.warning("۴ نقطه کافی است. برای شروع دوباره، ابتدا نقاط را پاک کنید.")
            else:
                st.session_state.calibration_points.append(
                    {
                        "pixel": (float(click["x"]), float(click["y"])),
                        "real_world": (float(real_x), float(real_y)),
                    }
                )
                st.rerun()
    else:
        st.write("روی تصویر کلیک کنید تا یک نقطه انتخاب شود.")

st.subheader(f"نقاط انتخاب‌شده ({len(st.session_state.calibration_points)}/4)")
st.dataframe(
    [
        {
            "شماره": i + 1,
            "پیکسل X": p["pixel"][0],
            "پیکسل Y": p["pixel"][1],
            "واقعی X (متر)": p["real_world"][0],
            "واقعی Y (متر)": p["real_world"][1],
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

    st.subheader("پیش‌نمایش نمای بالا (برای بررسی صحت کالیبراسیون)")
    st.image(
        cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
        caption="اگر خطوط خیابان/جدول در این تصویر مستقیم و موازی به‌نظر می‌رسند، کالیبراسیون درست است.",
        use_column_width=True,
        
    )

    if st.button("ذخیره کالیبراسیون", type="primary"):
        save_calibration(camera_id, matrix)
        st.success(f"کالیبراسیون دوربین «{camera_id}» ذخیره شد.")
else:
    st.info("برای محاسبه کالیبراسیون، ۴ نقطه لازم است.")
