from typing import Dict, List

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu") -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.vehicle_class_ids = [
            class_id
            for class_id, name in self.model.names.items()
            if name in VEHICLE_CLASS_NAMES
        ]
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False, device=self.device)

    def detect(self, image_bgr: np.ndarray, conf: float = 0.35, iou: float = 0.45) -> List[Dict]:
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            classes=self.vehicle_class_ids or None,
            verbose=False,
        )
        result = results[0]
        detections: List[Dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confs[i]),
                }
            )
        return detections