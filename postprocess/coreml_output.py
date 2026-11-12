"""Turn the outputs of a Core ML YOLO model exported with NMS (ultralytics `format=coreml, nms=True`) into detections.

Such a model returns `confidence` with shape (N, num_classes) (one score per class) and `coordinates` with shape (N, 4)
holding normalised (0..1) box centre x, centre y, width and height. Older single-score layouts (N,) are accepted too.
"""
import numpy as np


def parse_coreml_detections(coordinates, confidence, width, height, threshold):
    """Return (boxes, scores, class_ids): boxes as [x1, y1, x2, y2] in pixels of a width x height image.

    Detections whose best class score is below `threshold` are dropped.
    """
    coords = np.asarray(coordinates, dtype=float).reshape(-1, 4)
    conf = np.asarray(confidence, dtype=float)
    if conf.ndim == 2:
        class_ids = conf.argmax(axis=1) if conf.size else np.zeros(0, dtype=int)
        scores = conf.max(axis=1) if conf.size else np.zeros(0)
    else:
        scores = conf.reshape(-1)
        class_ids = np.zeros(len(scores), dtype=int)
    keep = scores >= threshold
    cx, cy, bw, bh = coords[keep].T
    boxes = np.stack([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height], axis=1) if keep.any() else np.zeros((0, 4))
    return boxes, scores[keep], class_ids[keep].astype(int)
