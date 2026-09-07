import numpy as np

from app.tracker import Sort, iou_batch


def test_iou_batch_perfect_overlap():
    boxes = np.array([[0, 0, 10, 10]])
    result = iou_batch(boxes, boxes)
    assert abs(result[0, 0] - 1.0) < 1e-6


def test_iou_batch_no_overlap():
    boxes_a = np.array([[0, 0, 10, 10]])
    boxes_b = np.array([[100, 100, 110, 110]])
    result = iou_batch(boxes_a, boxes_b)
    assert result[0, 0] == 0.0


def test_iou_batch_empty_inputs():
    result = iou_batch(np.empty((0, 4)), np.empty((0, 4)))
    assert result.shape == (0, 0)


def test_tracker_assigns_consistent_id_to_moving_object():
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    ids = []
    for frame in range(6):
        x = 100 + frame * 5
        detections = np.array([[x, 100, x + 80, 160, 0.9]])
        tracks = tracker.update(detections)
        if len(tracks):
            ids.append(int(tracks[0][4]))
    assert len(ids) > 0
    assert len(set(ids)) == 1


def test_tracker_survives_short_occlusion():
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    ids = []
    for frame in range(10):
        if 3 <= frame <= 4:
            detections = np.empty((0, 5))
        else:
            x = 100 + frame * 5
            detections = np.array([[x, 100, x + 80, 160, 0.9]])
        tracks = tracker.update(detections)
        if len(tracks):
            ids.append(int(tracks[0][4]))
    assert len(set(ids)) == 1


def test_tracker_assigns_distinct_ids_to_two_objects():
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    for frame in range(6):
        detections = np.array(
            [
                [100 + frame * 5, 100, 180 + frame * 5, 160, 0.9],
                [400 - frame * 3, 100, 480 - frame * 3, 160, 0.85],
            ]
        )
        tracks = tracker.update(detections)
    assert len(tracks) == 2
    assert tracks[0][4] != tracks[1][4]


def test_tracker_drops_track_after_max_age():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.3)
    tracker.update(np.array([[100, 100, 180, 160, 0.9]]))
    for _ in range(5):
        tracks = tracker.update(np.empty((0, 5)))
    assert len(tracker.trackers) == 0
