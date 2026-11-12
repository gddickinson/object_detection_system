"""The simple IoU tracker (postprocess/iou_tracker.py) and ObjectTracker's use of it."""
import numpy as np
import pytest

from postprocess.iou_tracker import associate_simple, box_iou


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0  # touching edges
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0  # zero union


def test_associate_matches_same_class_only():
    tracks, nid = {}, 1
    act, nid = associate_simple(tracks, np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), np.array([.9, .8]),
                                np.array([0, 1]), 1, nid, 30, 0.5)
    assert act == [1, 2] and nid == 3
    # frame 2: box 1 moves slightly (IoU 0.82); class-1 object reappears as class 0 -> new track
    act, nid = associate_simple(tracks, np.array([[1, 0, 11, 10], [50, 50, 60, 60]]), np.array([.9, .8]),
                                np.array([0, 0]), 2, nid, 30, 0.5)
    assert act == [1, 3] and nid == 4
    assert tracks[1]['last_frame'] == 2 and len(tracks[1]['boxes']) == 2
    assert tracks[2]['last_frame'] == 1


def test_threshold_is_strict_and_buffer_expires():
    tracks, nid = {}, 1
    _, nid = associate_simple(tracks, [[0, 0, 10, 10]], [1.0], [0], 1, nid, 1, 0.5)
    act, nid = associate_simple(tracks, [[5, 0, 15, 10]], [1.0], [0], 2, nid, 1, 1 / 3)  # IoU == thresh: no match
    assert act == [2]
    act, nid = associate_simple(tracks, [[5, 0, 15, 10]], [1.0], [0], 4, nid, 1, 0.1)  # track 2 last seen 2 > buffer 1
    assert act == [3]


def test_object_tracker_uses_it():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from models.tracker import ObjectTracker
    t = ObjectTracker({"models": {"tracker": {"type": "simple", "track_buffer": 5, "match_thresh": 0.3}}})
    d = {"boxes": np.array([[0., 0, 10, 10]]), "scores": np.array([.9]), "classes": np.array([2])}
    assert t.update(d)["active_tracks"] == [1]
    d["boxes"] = np.array([[1., 0, 11, 10]])
    r = t.update(d)
    assert r["active_tracks"] == [1] and len(r["tracks"][1]["boxes"]) == 2
    assert t._calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
