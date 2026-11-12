"""ObjectTracker with config.yaml settings on 6 synthetic moving objects (regression: the ByteTrack import path did not
exist in ultralytics 8.x, and the fallback IoU tracker used match_thresh 0.8 as a minimum IoU, so 6 objects produced
about 200 tracks)."""
import numpy as np
import pytest

from web.catalog import load_config
from web.synthetic import make_sequence

pytest.importorskip("torch")
pytest.importorskip("cv2")


def _run(cfg):
    from models.tracker import ObjectTracker
    t = ObjectTracker(cfg)
    seq = make_sequence(seed=1, n_objects=6, n_frames=60, speed=4.0, jitter=2.0, miss_rate=0.1)
    ids = set()
    for f in seq["frames"]:
        d = {"boxes": np.array(f["boxes"], float).reshape(-1, 4), "scores": np.array(f["scores"], float),
             "classes": np.array(f["classes"], int)}
        ids.update(t.update(d)["active_tracks"])
    return t, ids


def test_bytetrack_created_and_tracks_six_objects():
    pytest.importorskip("ultralytics")
    t, ids = _run(load_config())
    assert t.tracker is not None and t.tracker_type == "bytetrack"
    assert 6 <= len(ids) <= 8


def test_simple_fallback_default_iou():
    cfg = load_config()
    cfg["models"]["tracker"] = {k: v for k, v in cfg["models"]["tracker"].items() if k != "simple_iou_thresh"}
    cfg["models"]["tracker"]["type"] = "simple"
    t, ids = _run(cfg)
    assert t.tracker is None and t.simple_iou_thresh == 0.3
    assert 6 <= len(ids) <= 8


def test_simple_iou_thresh_configurable():
    cfg = load_config()
    cfg["models"]["tracker"].update(type="simple", simple_iou_thresh=0.8)
    t, ids = _run(cfg)
    assert t.simple_iou_thresh == 0.8 and len(ids) > 50
