"""create_detector without Core ML (regression: config.yaml sets detector type "coreml"; the fallback ObjectDetector kept
that type, logged "Unsupported model type: coreml" and returned no detections on every non-Apple-Silicon machine)."""
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")


def test_fallback_uses_yolov8():
    from models import detector as d
    cfg = {"models": {"detector": {"type": "coreml", "coreml_model_path": "x.mlpackage", "confidence": 0.4}}}
    with patch.object(d.ObjectDetector, "_init_model", return_value="model"):
        det = d.create_detector(cfg, {"os": "Linux", "architecture": "x86_64"})
    assert isinstance(det, d.ObjectDetector)
    assert det.model_type == "yolov8" and det.model_name == "yolov8n.pt"
    assert det.confidence == 0.4
    assert cfg["models"]["detector"]["type"] == "coreml"  # the caller's config is not modified


def test_yolov8_config_unchanged():
    from models import detector as d
    cfg = {"models": {"detector": {"type": "yolov8", "model_name": "yolov8s.pt"}}}
    with patch.object(d.ObjectDetector, "_init_model", return_value="model"):
        det = d.create_detector(cfg, {"os": "Linux", "architecture": "x86_64"})
    assert det.model_type == "yolov8" and det.model_name == "yolov8s.pt"


def test_missing_coreml_package_falls_back(tmp_path):
    """On Apple Silicon with coremltools but no .mlpackage on disk (weights are not committed), use YOLOv8."""
    import sys
    from unittest.mock import MagicMock
    from models import detector as d
    cfg = {"models": {"detector": {"type": "coreml", "coreml_model_path": str(tmp_path / "missing.mlpackage")}}}
    with patch.dict(sys.modules, {"coremltools": MagicMock()}), \
            patch.object(d.ObjectDetector, "_init_model", return_value="model"):
        det = d.create_detector(cfg, {"os": "Darwin", "architecture": "arm64"})
    assert isinstance(det, d.ObjectDetector) and det.model_type == "yolov8"
