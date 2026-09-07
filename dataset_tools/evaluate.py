import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO, settings

settings.update({"wandb": False})

PRECISION_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.80
MAP50_THRESHOLD = 0.80
MAX_FRAME_TIME_SECONDS = 3.0


def load_test_images(data_yaml_path):
    import yaml

    with open(data_yaml_path) as f:
        config = yaml.safe_load(f)

    base_path = config.get("path", os.path.dirname(data_yaml_path))
    test_rel = config.get("test", "images/test")
    test_dir = os.path.join(base_path, test_rel)

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(test_dir, ext)))
    return sorted(images)


def measure_inference_time(model, image_paths, device):
    if not image_paths:
        return None
    durations = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        start = time.perf_counter()
        model.predict(source=image, device=device, verbose=False)
        durations.append(time.perf_counter() - start)
    if not durations:
        return None
    return sum(durations) / len(durations), max(durations)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned YOLOv8 model against the project's acceptance criteria."
    )
    parser.add_argument("weights", help="Path to the trained model weights (best.pt).")
    parser.add_argument("data_yaml", help="Path to the dataset's data.yaml.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.weights)

    metrics = model.val(data=args.data_yaml, split="test", device=args.device, verbose=False)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)

    test_images = load_test_images(args.data_yaml)
    timing = measure_inference_time(model, test_images, args.device)

    print("=== Detection metrics ===")
    print(f"Precision: {precision:.3f}  (target >= {PRECISION_THRESHOLD})  {'PASS' if precision >= PRECISION_THRESHOLD else 'FAIL'}")
    print(f"Recall:    {recall:.3f}  (target >= {RECALL_THRESHOLD})  {'PASS' if recall >= RECALL_THRESHOLD else 'FAIL'}")
    print(f"mAP@0.5:   {map50:.3f}  (target >= {MAP50_THRESHOLD})  {'PASS' if map50 >= MAP50_THRESHOLD else 'FAIL'}")

    print("=== Inference timing ===")
    if timing is None:
        print("No test images found to time.")
    else:
        avg_seconds, max_seconds = timing
        print(f"Average: {avg_seconds:.3f}s  Max: {max_seconds:.3f}s  (target < {MAX_FRAME_TIME_SECONDS}s per frame)")
        print("PASS" if max_seconds < MAX_FRAME_TIME_SECONDS else "FAIL")


if __name__ == "__main__":
    main()