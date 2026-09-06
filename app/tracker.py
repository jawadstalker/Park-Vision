import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    area = max(w * h, 1e-6)
    aspect = w / max(h, 1e-6)
    return np.array([cx, cy, area, aspect])


def z_to_bbox(z):
    cx, cy, area, aspect = z[0], z[1], z[2], z[3]
    area = max(area, 1e-6)
    w = np.sqrt(area * aspect)
    h = area / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou_batch(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.expand_dims(boxes_a, 1)
    boxes_b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    intersection = inter_w * inter_h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - intersection

    return intersection / np.maximum(union, 1e-6)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.state = np.zeros(7)
        self.state[:4] = bbox_to_z(bbox)

        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 100.0

        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.1

        self.R = np.eye(4) * 1.0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.state = self.F @ self.state
        if self.state[2] + self.state[6] <= 0:
            self.state[6] = 0.0
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return z_to_bbox(self.state[:4])

    def update(self, bbox):
        z = bbox_to_z(bbox)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self):
        return z_to_bbox(self.state[:4])


def associate_detections_to_trackers(detections, tracked_boxes, iou_threshold):
    if len(tracked_boxes) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = iou_batch(detections, tracked_boxes)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_detections = []
    unmatched_trackers = list(range(len(tracked_boxes)))

    matched_tracker_indices = set()
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matches.append((row, col))
            matched_tracker_indices.add(col)
        else:
            unmatched_detections.append(row)

    for row in range(len(detections)):
        if row not in [m[0] for m in matches] and row not in unmatched_detections:
            unmatched_detections.append(row)

    unmatched_trackers = [i for i in unmatched_trackers if i not in matched_tracker_indices]

    return matches, unmatched_detections, unmatched_trackers


class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1

        predicted_boxes = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))
        detection_boxes = detections[:, :4] if len(detections) else np.empty((0, 4))

        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
            detection_boxes, predicted_boxes, self.iou_threshold
        )

        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detection_boxes[det_idx])

        for det_idx in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(detection_boxes[det_idx]))

        output = []
        alive_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update <= self.max_age:
                alive_trackers.append(tracker)
                is_confirmed = tracker.hits >= self.min_hits or self.frame_count <= self.min_hits
                if tracker.time_since_update == 0 and is_confirmed:
                    bbox = tracker.get_bbox()
                    output.append([bbox[0], bbox[1], bbox[2], bbox[3], tracker.id])

        self.trackers = alive_trackers

        return np.array(output) if output else np.empty((0, 5))
