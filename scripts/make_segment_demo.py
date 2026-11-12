"""Render a synthetic scene and segment it with the repo's ObjectSegmentor (OpenCV fallback, no SAM checkpoint).

Writes web/static/demo/segment_scene.png and web/demo/segment_demo.json (boxes, contours, mask areas). Needs the
desktop app's dependencies (torch, opencv-python). Run: python scripts/make_segment_demo.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.segmentor import ObjectSegmentor  # noqa: E402

PNG = ROOT / "web/static/demo/segment_scene.png"
OUT = ROOT / "web/demo/segment_demo.json"
W, H, PAD = 640, 400, 10


def scene():
    """Textured background plus five filled shapes; returns (BGR image, [(name, box)])."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 170 + 40 * (xx / W) - 30 * (yy / H)
    img = np.clip(np.dstack([base, base + 8, base - 10]) + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    shapes = []

    def add(name, mask, color):
        img[mask > 0] = color
        ys, xs = np.nonzero(mask)
        shapes.append((name, [int(max(xs.min() - PAD, 0)), int(max(ys.min() - PAD, 0)),
                              int(min(xs.max() + PAD, W - 1)), int(min(ys.max() + PAD, H - 1))]))

    def blank():
        return np.zeros((H, W), np.uint8)
    m = blank()
    cv2.ellipse(m, (120, 110), (70, 40), 20, 0, 360, 255, -1)
    add("ellipse", m, (60, 70, 160))
    m = blank()
    cv2.fillPoly(m, [cv2.boxPoints(((330, 120), (120, 60), -25)).astype(np.int32)], 255)
    add("rotated rectangle", m, (150, 60, 40))
    m = blank()
    cv2.circle(m, (520, 110), 55, 255, -1)
    add("disc", m, (40, 130, 60))
    m = blank()
    cv2.fillPoly(m, [np.array([[90, 360], [190, 220], [260, 370]], np.int32)], 255)
    add("triangle", m, (30, 30, 30))
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    r = 70 + 18 * np.sin(3 * t) + 10 * np.cos(5 * t)
    m = blank()
    cv2.fillPoly(m, [np.c_[460 + r * np.cos(t), 290 + 0.8 * r * np.sin(t)].astype(np.int32)], 255)
    add("blob", m, (120, 40, 140))
    return img, shapes


def build():
    """Return (image, demo dict) computed by the repo's segmentor."""
    img, shapes = scene()
    seg = ObjectSegmentor({"models": {"segmentor": {"type": "sam", "checkpoint": "no_such_checkpoint.pth"}}})
    assert seg.sam is None, "expected the OpenCV fallback (no SAM checkpoint)"
    res = seg.segment(img, np.array([b for _, b in shapes], dtype=float))
    objects = []
    for (name, box), mask, cnt in zip(shapes, res["masks"], res["contours"]):
        objects.append({"name": name, "box": box, "mask_area": int(mask.sum()),
                        "box_area": (box[2] - box[0]) * (box[3] - box[1]),
                        "contour": cnt.reshape(-1, 2).tolist() if cnt is not None else []})
    return img, {"image": "static/demo/segment_scene.png", "width": W, "height": H,
                 "method": "ObjectSegmentor._basic_segment (OpenCV fallback)", "objects": objects}


def main():
    img, demo = build()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PNG), img)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(demo, separators=(",", ":")) + "\n")
    print(f"wrote {PNG.name} and {OUT.name}: "
          + ", ".join(f"{o['name']} {len(o['contour'])} pts" for o in demo["objects"]))


if __name__ == "__main__":
    main()
