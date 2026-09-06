import argparse
import json

from app.calibration import load_calibration
from app.vehicle_detector import VehicleDetector
from app.video_processor import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Run vehicle detection, tracking, and gap detection on a recorded street video."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("camera_id", help="Calibrated camera id (see /calibrate).")
    parser.add_argument("output", help="Path to write per-frame results as JSON Lines.")
    parser.add_argument("--gap-threshold-m", type=float, default=4.5)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    matrix = load_calibration(args.camera_id)
    if matrix is None:
        raise SystemExit(
            f"Camera '{args.camera_id}' is not calibrated. Call /calibrate first."
        )

    detector = VehicleDetector(model_path=args.model, device=args.device)

    frame_count = 0
    with open(args.output, "w") as f:
        for result in process_video(
            args.video, detector, matrix, gap_threshold_m=args.gap_threshold_m, conf=args.conf
        ):
            serializable = {
                "frame_index": result["frame_index"],
                "timestamp_s": result["timestamp_s"],
                "tracking_gap_time_ms": result["tracking_gap_time_ms"],
                "vehicles": [
                    {"bbox": v["bbox"], "track_id": v["track_id"]} for v in result["vehicles"]
                ],
                "spots": [
                    {
                        "id": s["id"],
                        "status": s["status"],
                        "start_m": s["start_m"],
                        "end_m": s["end_m"],
                        "length_m": s["length_m"],
                    }
                    for s in result["spots"]
                ],
            }
            f.write(json.dumps(serializable) + "\n")
            frame_count += 1

    print(f"Processed {frame_count} frames, wrote results to {args.output}")


if __name__ == "__main__":
    main()
