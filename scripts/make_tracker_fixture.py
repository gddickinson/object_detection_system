"""Write tests/fixtures/tracker_cases.json: Python results that web/static/track_engine.test.mjs must reproduce.

IoU cases, class colours (postprocess/colors.py) and the simple IoU tracker (postprocess/iou_tracker.py via
web/synthetic.run_tracker) on synthetic sequences. Run: python scripts/make_tracker_fixture.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from postprocess.colors import class_color_bgr  # noqa: E402
from postprocess.iou_tracker import box_iou  # noqa: E402
from web.synthetic import make_sequence, run_tracker  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "tracker_cases.json"
IOU_PAIRS = [([0, 0, 10, 10], [0, 0, 10, 10]), ([0, 0, 10, 10], [5, 0, 15, 10]), ([0, 0, 10, 10], [20, 20, 30, 30]),
             ([0, 0, 10, 10], [10, 0, 20, 10]), ([0, 0, 0, 0], [0, 0, 0, 0]), ([1.5, 2.25, 40, 30], [10, 5.5, 60.2, 44])]
NAMES = ["person", "cat", "potted plant", "traffic light", "teddy bear", "toothbrush", "dog", "car"]
SEQS = [dict(seed=1), dict(seed=2, n_objects=10, speed=8.0, miss_rate=0.25, n_frames=40),
        dict(seed=3, n_objects=4, speed=2.0, jitter=5.0, miss_rate=0.0, n_frames=50)]
SETTINGS = [(30, 0.8), (30, 0.3), (2, 0.5), (0, 0.1)]


def build():
    cases = []
    for kw in SEQS:
        seq = make_sequence(**kw)
        cases.append({"sequence": seq, "runs": [run_tracker(seq["frames"], b, t) for b, t in SETTINGS]})
    return {"iou": [{"a": a, "b": b, "iou": box_iou(a, b)} for a, b in IOU_PAIRS],
            "colors": [{"name": n, "bgr": list(class_color_bgr(n))} for n in NAMES],
            "tracker": cases}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
