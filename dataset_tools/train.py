import argparse

from ultralytics import YOLO, settings

settings.update({"wandb": False})


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLOv8 vehicle detector on the street-parking dataset."
    )
    parser.add_argument("data_yaml", help="Path to the dataset's data.yaml (from organize_dataset.py).")
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', '0,1', etc.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="street_parking")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()