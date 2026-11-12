"""Demo results for the static site: the simple IoU tracker on synthetic sequences (computed on request by
postprocess/iou_tracker.py) and the OpenCV-fallback segmentation of a synthetic scene (precomputed by
scripts/make_segment_demo.py, which needs torch)."""
import json
from pathlib import Path

from .catalog import load_config
from .synthetic import DEFAULTS, make_sequence, run_tracker

SEGMENT_JSON = Path(__file__).resolve().parent / "demo" / "segment_demo.json"


def tracker_settings():
    t = load_config().get("models", {}).get("tracker", {})
    return {"type": t.get("type", "bytetrack"), "track_buffer": t.get("track_buffer", 30),
            "match_thresh": t.get("match_thresh", 0.8), "simple_iou_thresh": t.get("simple_iou_thresh", 0.3)}


def tracker_demo():
    """One synthetic sequence (DEFAULTS) tracked with config.yaml's simple_iou_thresh and a few other IoU thresholds."""
    seq = make_sequence(**DEFAULTS)
    cfg = tracker_settings()
    runs = [run_tracker(seq["frames"], cfg["track_buffer"], th) for th in (cfg["simple_iou_thresh"], 0.8, 0.5, 0.1)]
    return {"sequence": seq, "config": cfg, "runs": runs}


def segment_demo():
    return json.loads(SEGMENT_JSON.read_text(encoding="utf-8"))
