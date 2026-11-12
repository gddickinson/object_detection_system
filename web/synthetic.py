"""Synthetic detection sequences for the tracker demo and the JS-vs-Python fixture (numpy only).

Not real detector output: objects of a few COCO classes move at constant velocity and bounce off the frame edges; each
frame's 'detections' are their boxes with Gaussian corner jitter, a random score, and a chance of being missed. Because
the true object behind every detection is known, the tracker's identity changes can be counted.
"""
import numpy as np

from postprocess.iou_tracker import associate_simple

# (COCO class id, name, box width, box height) in pixels of a 640 x 360 frame
KINDS = [(0, "person", 34, 80), (2, "car", 90, 46), (16, "dog", 52, 36), (1, "bicycle", 60, 44)]
DEFAULTS = {"n_objects": 6, "n_frames": 60, "width": 640, "height": 360, "speed": 4.0, "jitter": 2.0,
            "miss_rate": 0.1, "seed": 1}


def make_sequence(n_objects=6, n_frames=60, width=640, height=360, speed=4.0, jitter=2.0, miss_rate=0.1, seed=1):
    """Return {params, frames: [{boxes, scores, classes, gt}]} with plain-Python numbers (rounded to 0.01 px)."""
    rng = np.random.default_rng(seed)
    kinds = [KINDS[i % len(KINDS)] for i in range(n_objects)]
    size = np.array([[k[2], k[3]] for k in kinds], dtype=float)
    pos = rng.uniform([0, 0], [1, 1], (n_objects, 2)) * (np.array([width, height]) - size)
    ang = rng.uniform(0, 2 * np.pi, n_objects)
    vel = np.c_[np.cos(ang), np.sin(ang)] * speed * rng.uniform(0.5, 1.5, (n_objects, 1))
    frames = []
    for _ in range(n_frames):
        pos += vel
        for d, lim in ((0, width), (1, height)):
            lo, hi = pos[:, d] < 0, pos[:, d] + size[:, d] > lim
            vel[lo | hi, d] *= -1
            pos[:, d] = np.clip(pos[:, d], 0, lim - size[:, d])
        seen = rng.random(n_objects) >= miss_rate
        noise = rng.normal(0, jitter, (n_objects, 4))
        scores = rng.uniform(0.5, 0.95, n_objects)
        f = {"boxes": [], "scores": [], "classes": [], "gt": []}
        for i in np.flatnonzero(seen):
            x1, y1 = pos[i]
            box = np.array([x1, y1, x1 + size[i, 0], y1 + size[i, 1]]) + noise[i]
            f["boxes"].append([round(float(v), 2) for v in box])
            f["scores"].append(round(float(scores[i]), 3))
            f["classes"].append(kinds[i][0])
            f["gt"].append(int(i))
        frames.append(f)
    params = dict(n_objects=n_objects, n_frames=n_frames, width=width, height=height, speed=speed, jitter=jitter,
                  miss_rate=miss_rate, seed=seed)
    return {"params": params, "names": {str(k[0]): k[1] for k in KINDS}, "frames": frames}


def run_tracker(frames, track_buffer=30, match_thresh=0.8):
    """Run the repo's simple IoU tracker over a sequence, as ObjectTracker.update does frame by frame.

    Returns per frame the active track ids and the track id given to each detection, plus totals: tracks started and
    identity changes (a true object's track id differs from the one it had when last detected).
    """
    tracks, next_id, out = {}, 1, []
    last_id, switches = {}, 0
    for frame_count, f in enumerate(frames, start=1):
        boxes = [list(b) for b in f["boxes"]]
        active, next_id = associate_simple(tracks, boxes, f["scores"], f["classes"], frame_count, next_id,
                                           track_buffer, match_thresh)
        det_track = [next(t for t in active if tracks[t]["boxes"][-1] is b) for b in boxes]
        for g, t in zip(f.get("gt", []), det_track):
            if g in last_id and last_id[g] != t:
                switches += 1
            last_id[g] = t
        out.append({"active": active, "det_track": det_track})
    return {"track_buffer": track_buffer, "match_thresh": match_thresh, "frames": out,
            "n_tracks": next_id - 1, "id_changes": switches}
