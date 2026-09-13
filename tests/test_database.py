from app.database import get_history, list_cameras_with_history, save_detection


def sample_vehicles():
    return [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9}]


def sample_spots():
    return [
        {"id": 1, "status": "occupied", "start_m": 0.0, "end_m": 4.0, "length_m": 4.0},
        {"id": 2, "status": "empty", "start_m": 4.0, "end_m": 8.0, "length_m": 4.0},
    ]


def test_save_and_get_history_roundtrip():
    save_detection(
        camera_id="cam-a",
        mode="calibrated",
        frame_width=640,
        frame_height=480,
        vehicles=sample_vehicles(),
        spots=sample_spots(),
        processing_time_ms=12.5,
    )
    records = get_history("cam-a")
    assert len(records) == 1
    record = records[0]
    assert record["camera_id"] == "cam-a"
    assert record["mode"] == "calibrated"
    assert record["vehicle_count"] == 1
    assert record["occupied_count"] == 1
    assert record["empty_count"] == 1
    assert record["vehicles"] == sample_vehicles()
    assert record["spots"] == sample_spots()


def test_get_history_empty_for_unknown_camera():
    assert get_history("never-seen") == []


def test_get_history_orders_most_recent_first():
    for i in range(3):
        save_detection(
            camera_id="cam-b",
            mode="pixel",
            frame_width=640,
            frame_height=480,
            vehicles=[],
            spots=[],
            processing_time_ms=float(i),
        )
    records = get_history("cam-b")
    assert [r["processing_time_ms"] for r in records] == [2.0, 1.0, 0.0]


def test_get_history_respects_limit():
    for i in range(5):
        save_detection(
            camera_id="cam-c",
            mode="pixel",
            frame_width=640,
            frame_height=480,
            vehicles=[],
            spots=[],
            processing_time_ms=float(i),
        )
    records = get_history("cam-c", limit=2)
    assert len(records) == 2


def test_list_cameras_with_history_returns_distinct_sorted_ids():
    save_detection("cam-z", "pixel", 1, 1, [], [], 0.0)
    save_detection("cam-a", "pixel", 1, 1, [], [], 0.0)
    save_detection("cam-a", "pixel", 1, 1, [], [], 0.0)
    assert list_cameras_with_history() == ["cam-a", "cam-z"]
