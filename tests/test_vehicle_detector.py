import os

from PIL import Image

from app.vehicle_detector import VehicleDetector


def test_base_coco_model_resolves_known_vehicle_class_ids():
    detector = VehicleDetector(model_path="yolov8n.pt", device="cpu")
    assert set(detector.vehicle_class_ids) == {2, 3, 5, 7}


def test_finetuned_single_class_model_resolves_its_own_class_id(tmp_path):
    from ultralytics import YOLO, settings

    settings.update({"wandb": False})

    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(4):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(images_dir / f"img{i}.jpg")
        (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['car']\n"
    )

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=64,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="test",
        verbose=False,
    )

    weights_path = tmp_path / "runs" / "test" / "weights" / "best.pt"
    assert weights_path.exists()

    detector = VehicleDetector(model_path=str(weights_path), device="cpu")

    # Before the fix, this stayed hardcoded to the COCO ids {2, 3, 5, 7},
    # which don't exist in a single-class fine-tuned model and silently
    # filtered out every detection.
    assert detector.vehicle_class_ids == [0]