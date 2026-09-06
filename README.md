# Smart Parking Gap Detection API

## Structure

```
app/
  main.py             FastAPI app, loads YOLOv8 on startup
  routes.py           POST /detect, POST /calibrate, GET /status
  schemas.py          Pydantic request/response models
  vehicle_detector.py YOLOv8 wrapper, returns vehicle bounding boxes
  perspective.py      Pixel <-> real-world coordinate helpers
  calibration.py      Per-camera homography storage (JSON files)
  gap_detector.py      Core Gap Detection algorithm
requirements.txt
```

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Calibrate a camera

Send 4 pixel points and their corresponding real-world coordinates (in meters,
measured along the street) for a given `camera_id`:

```
POST /calibrate
{
  "camera_id": "street-01",
  "points": [
    {"pixel": {"x": 120, "y": 480}, "real_world": {"x": 0, "y": 0}},
    {"pixel": {"x": 640, "y": 480}, "real_world": {"x": 20, "y": 0}},
    {"pixel": {"x": 640, "y": 200}, "real_world": {"x": 20, "y": 5}},
    {"pixel": {"x": 120, "y": 200}, "real_world": {"x": 0, "y": 5}}
  ]
}
```

## Detect

```
POST /detect  (multipart/form-data)
  camera_id: street-01
  gap_threshold_m: 4.5
  image: <file>
```

Returns detected vehicles (pixel bbox + confidence) and parking spots
(occupied/empty, with real-world start/end/length in meters) computed by
sorting vehicles along the street axis and measuring the gap between
consecutive vehicle edges after perspective transform.

## Next steps

- Replace `yolov8n.pt` with a model fine-tuned on the project's own
  street-parking dataset (COCO pretrain + collected Mashhad footage).
- Add vehicle tracking (SORT/DeepSORT) across frames for video input.
- Add lane/no-parking-zone masks to filter out gaps that aren't legal
  parking spots (bus stops, driveways).
