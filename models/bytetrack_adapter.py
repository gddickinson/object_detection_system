"""ByteTrack from ultralytics 8.x (ultralytics.trackers.byte_tracker.BYTETracker) behind a plain numpy interface.

BYTETracker takes an args namespace (the fields of ultralytics' cfg/trackers/bytetrack.yaml) and, per frame, a
Boxes-like object with conf, cls and xywh; it returns rows [x1, y1, x2, y2, track_id, score, cls, det_index].
"""
from types import SimpleNamespace

import numpy as np


def create_bytetrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30):
    """A BYTETracker; track_thresh is the high-score (first-stage) and new-track threshold, match_thresh the
    association threshold of ultralytics' matching (1 - IoU cost). Raises ImportError without ultralytics."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=track_thresh, track_low_thresh=0.1,
                           new_track_thresh=track_thresh, track_buffer=track_buffer, match_thresh=match_thresh,
                           fuse_score=True)
    try:
        return BYTETracker(args, frame_rate=frame_rate)  # ultralytics < 8.4
    except TypeError:
        return BYTETracker(args)  # newer releases take the frame rate from args only


def bytetrack_update(tracker, boxes, scores, classes, image_shape=(1, 1)):
    """Feed one frame of detections (xyxy boxes, scores, class ids) and return the (N, 8) output rows."""
    from ultralytics.engine.results import Boxes
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    data = np.concatenate([boxes, np.asarray(scores, np.float32).reshape(-1, 1),
                           np.asarray(classes, np.float32).reshape(-1, 1)], axis=1)
    return tracker.update(Boxes(data, image_shape))
