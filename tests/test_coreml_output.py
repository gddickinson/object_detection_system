"""Core ML YOLO output parsing (regression: the detector treated the (N, 80) class-score array as one score per box,
raised "truth value of an array is ambiguous" and returned no detections at all)."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from postprocess.coreml_output import parse_coreml_detections

# Two detections from the repo's yolov8n.mlpackage layout: per-class scores and normalised centre xywh.
CONF = np.zeros((2, 80))
CONF[0, 15] = 0.77
CONF[1, 58] = 0.40
COORDS = np.array([[0.25, 0.5, 0.5, 0.2], [0.8, 0.25, 0.4, 0.3]])


def test_parse_per_class_scores():
    boxes, scores, classes = parse_coreml_detections(COORDS, CONF, 1000, 2000, 0.25)
    assert classes.tolist() == [15, 58]
    assert np.allclose(scores, [0.77, 0.40])
    assert np.allclose(boxes[0], [0, 800, 500, 1200])
    assert np.allclose(boxes[1], [600, 200, 1000, 800])


def test_threshold_and_empty():
    boxes, scores, classes = parse_coreml_detections(COORDS, CONF, 1000, 2000, 0.5)
    assert classes.tolist() == [15] and boxes.shape == (1, 4)
    boxes, scores, classes = parse_coreml_detections(np.zeros((0, 4)), np.zeros((0, 80)), 10, 10, 0.5)
    assert boxes.shape == (0, 4) and len(scores) == 0


def test_coreml_detector_returns_detections():
    """The real CoreMLDetector.detect with a stubbed Core ML model (needs torch + OpenCV + Pillow to import)."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from models.detector import CoreMLDetector
    det = CoreMLDetector.__new__(CoreMLDetector)
    det.confidence = 0.25
    det.labels = {15: "cat", 58: "potted plant"}
    det.model = MagicMock()
    det.model.predict.return_value = {"coordinates": COORDS, "confidence": CONF}
    r = det.detect(np.zeros((2000, 1000, 3), dtype=np.uint8))
    assert r["class_names"] == ["cat", "potted plant"]
    assert np.allclose(r["boxes"][0], [0, 800, 500, 1200])
    assert np.allclose(r["centers"][1], [800, 500])
