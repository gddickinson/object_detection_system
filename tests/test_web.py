"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests -------------------------------------------------------------------------------------

def test_info_counts():
    j = client.get("/api/info").json()
    assert j["n_classes"] == 80 and j["n_detectors"] == 3 and j["n_trackers"] == 3
    assert "api_key" not in str(j["config"]) and "auth" not in j["config"]


def test_labels_colours_match_python():
    from postprocess.colors import class_color_bgr
    lab = client.get("/api/labels").json()
    assert lab[0]["name"] == "person" and lab[-1]["name"] == "toothbrush"
    b, g, r = class_color_bgr("person")
    assert lab[0]["color"] == f"#{r:02x}{g:02x}{b:02x}"


def test_modules_read_without_import():
    groups = client.get("/api/modules").json()
    files = {m["file"]: m for g in groups for m in g["modules"]}
    assert "models/tracker.py" in files
    assert any(c["name"] == "ObjectTracker" for c in files["models/tracker.py"]["classes"])


def test_tracker_demo_uses_config():
    j = client.get("/api/demo/tracker").json()
    assert j["config"]["simple_iou_thresh"] == j["runs"][0]["match_thresh"] == 0.3
    assert len(j["sequence"]["frames"]) == len(j["runs"][0]["frames"])
    assert j["runs"][-1]["n_tracks"] >= j["sequence"]["params"]["n_objects"]


def test_segment_demo_image_present():
    j = client.get("/api/demo/segment").json()
    assert (STATIC / "demo" / "segment_scene.png").exists() and j["image"] == "static/demo/segment_scene.png"
    assert len(j["objects"]) == 5 and all(len(o["contour"]) >= 3 for o in j["objects"])


def test_tracker_fixture_current():
    """tests/fixtures/tracker_cases.json still matches the Python code (rerun scripts/make_tracker_fixture.py)."""
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_tracker_fixture
    stored = json.loads((ROOT / "tests/fixtures/tracker_cases.json").read_text())
    fresh = json.loads(json.dumps(make_tracker_fixture.build()))
    for a, b in zip(stored["iou"], fresh["iou"]):
        assert a["iou"] == pytest.approx(b["iou"], rel=1e-9, abs=1e-12)
    assert stored["colors"] == fresh["colors"]
    for a, b in zip(stored["tracker"], fresh["tracker"]):
        for fa, fb in zip(a["sequence"]["frames"], b["sequence"]["frames"]):
            assert fa["gt"] == fb["gt"] and fa["classes"] == fb["classes"]
            assert sum(fa["boxes"], []) == pytest.approx(sum(fb["boxes"], []), abs=0.011)
        assert [r["n_tracks"] for r in a["runs"]] == [r["n_tracks"] for r in b["runs"]]


def test_segment_demo_current():
    """web/demo/segment_demo.json matches scripts/make_segment_demo.py (needs torch + OpenCV; skipped otherwise)."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_segment_demo
    _, fresh = make_segment_demo.build()
    stored = json.loads((ROOT / "web/demo/segment_demo.json").read_text())
    assert [o["box"] for o in stored["objects"]] == [o["box"] for o in fresh["objects"]]
    assert [o["mask_area"] for o in stored["objects"]] == pytest.approx([o["mask_area"] for o in fresh["objects"]], rel=0.02)


def test_status_and_upload_validation():
    assert "packages" in client.get("/api/status").json()
    assert client.post("/api/detect", data={"confidence": "0.5"}).status_code == 400
    assert client.post("/api/detect", data={"example": "cats", "confidence": "2"}).status_code == 400
    r = client.post("/api/detect_video", files={"file": ("a.txt", b"x")}, data={"confidence": "0.5"})
    assert r.status_code == 400


def test_detect_with_stub_models(monkeypatch):
    """POST /api/detect end to end with stand-in models (the real ones need torch)."""
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from web import inference

    class Det:
        model = object()
        def set_confidence(self, c): self.c = c
        def detect(self, img):
            return {"boxes": np.array([[1., 2, 11, 22]]), "scores": np.array([.9]), "classes": np.array([15]),
                    "class_names": ["cat"], "centers": np.array([[6., 12]])}

    class Seg:
        sam = None
        def segment(self, img, boxes):
            return {"masks": [np.zeros(img.shape[:2], bool)], "contours": [np.array([[[1, 2]], [[11, 2]], [[11, 22]]])]}

    monkeypatch.setattr(inference, "_models", lambda backend: (Det(), Seg()))
    monkeypatch.setattr(inference, "_repo_json", lambda results: {"detections": {"n": len(results["detections"]["boxes"])}})
    ok, png = cv2.imencode(".png", np.zeros((30, 40, 3), np.uint8))
    r = client.post("/api/detect", files={"file": ("x.png", png.tobytes())}, data={"confidence": "0.3"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["width"], j["height"]) == (40, 30) and j["detections"][0]["class_name"] == "cat"
    assert j["contours"] == [[[1, 2], [11, 2], [11, 22]]] and j["segmentor"] == "OpenCV fallback"
