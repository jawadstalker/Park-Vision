import argparse
import os

import cv2


def extract_frames(video_path, output_dir, street, lighting, interval_seconds, quality):
    os.makedirs(output_dir, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * interval_seconds)))

    frame_index = 0
    saved_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            filename = f"{street}_{lighting}_{saved_index:05d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved_index += 1
        frame_index += 1

    capture.release()
    return saved_index


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a street video for dataset collection."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("output_dir", help="Directory to save extracted frames.")
    parser.add_argument("--street", required=True, help="Street identifier, e.g. azadi-blvd.")
    parser.add_argument(
        "--lighting",
        required=True,
        choices=["day", "dusk", "night"],
        help="Lighting condition of the footage.",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Seconds between saved frames (default: 2.0)."
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95).")
    args = parser.parse_args()

    count = extract_frames(
        args.video, args.output_dir, args.street, args.lighting, args.interval, args.quality
    )
    print(f"Saved {count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
