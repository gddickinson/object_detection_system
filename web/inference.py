"""Run the repo's own models on an uploaded image or video (local app only; needs torch, ultralytics, opencv-python).

Models are created lazily by the desktop app's classes (models/detector.py, segmentor.py, tracker.py) with the settings
in config.yaml, and cached for the life of the server. Nothing here is imported until a request needs it, so the rest of
the web app (and the static build) runs without torch.
"""
import importlib.util
import json
import os
import tempfile
import threading
import time

from .catalog import ROOT, load_config

_LOCK = threading.Lock()
_CACHE = {}
BACKENDS = ("auto", "yolov8", "coreml")


class InferenceError(Exception):
    pass


def status():
    """Which optional packages and model files are available on this computer."""
    pk = {p: importlib.util.find_spec(p) is not None
          for p in ("torch", "cv2", "ultralytics", "coremltools", "segment_anything")}
    cfg = load_config().get("models", {})
    ckpt = (cfg.get("segmentor") or {}).get("checkpoint", "")
    files = {"data/models/yolov8n.pt": (ROOT / "data/models/yolov8n.pt").exists(),
             "data/models/yolov8n.mlpackage": (ROOT / "data/models/yolov8n.mlpackage").exists()}
    if ckpt:
        files[ckpt] = (ROOT / ckpt).exists()
    return {"packages": pk, "files": files, "ready": pk["torch"] and pk["cv2"] and pk["ultralytics"],
            "backends": list(BACKENDS)}


def _system_info():
    import platform
    return {"os": platform.system(), "architecture": platform.machine(), "acceleration": "CPU"}


def _models(backend):
    """(detector, segmentor) for a back end, created once. Paths in config.yaml are relative to the repo root."""
    if backend in _CACHE:
        return _CACHE[backend]
    if not status()["ready"]:
        raise InferenceError("Inference needs torch, ultralytics and opencv-python: pip install -r requirements.txt")
    cwd = os.getcwd()
    os.chdir(ROOT)  # the model classes resolve data/models/... relative to the working directory
    try:
        from models.detector import CoreMLDetector, ObjectDetector, create_detector
        from models.segmentor import ObjectSegmentor
        cfg = load_config()
        det_cfg = cfg.setdefault("models", {}).setdefault("detector", {})
        if backend == "auto":
            det = create_detector(cfg, _system_info())
        elif backend == "coreml":
            if importlib.util.find_spec("coremltools") is None:
                raise InferenceError("The Core ML back end needs coremltools (macOS).")
            det = CoreMLDetector(cfg, _system_info())
        else:
            det_cfg.update(type="yolov8", model_name=str(ROOT / "data/models/yolov8n.pt"))
            det = ObjectDetector(cfg, _system_info())
        seg = ObjectSegmentor(cfg, _system_info())
    finally:
        os.chdir(cwd)
    if getattr(det, "model", None) is None:
        hint = (" Export the Core ML model with: yolo export model=yolov8n.pt format=coreml nms=True"
                if backend == "coreml" else "")
        raise InferenceError(f"The {backend} detector could not load its model (see the server log).{hint}")
    _CACHE[backend] = (det, seg)
    return _CACHE[backend]


def _detections_json(d):
    out = []
    for i in range(len(d["boxes"])):
        box = [round(float(v), 2) for v in d["boxes"][i][:4]]
        out.append({"box": box, "score": round(float(d["scores"][i]), 4), "class_id": int(d["classes"][i]),
                    "class_name": str(d["class_names"][i]),
                    "center": [round(float(v), 2) for v in d["centers"][i]]})
    return out


def _repo_json(results):
    """The results file exactly as the desktop app saves it (utils/format_converter.save_results_to_json)."""
    from utils.format_converter import save_results_to_json
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "results.json")
        save_results_to_json(results, path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def detect_image(data: bytes, confidence: float, segment: bool, backend: str = "auto"):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InferenceError("Could not read that file as an image.")
    with _LOCK:
        det, seg = _models(backend)
        det.set_confidence(confidence)
        t0 = time.perf_counter()
        d = det.detect(image)
        t1 = time.perf_counter()
        s = seg.segment(image, d["boxes"]) if (segment and len(d["boxes"])) else {"masks": [], "contours": []}
        t2 = time.perf_counter()
    contours = [c.reshape(-1, 2).tolist() if c is not None else [] for c in s["contours"]]
    return {"width": int(image.shape[1]), "height": int(image.shape[0]), "backend": type(det).__name__,
            "segmentor": "SAM" if getattr(seg, "sam", None) is not None else "OpenCV fallback",
            "confidence": confidence, "detections": _detections_json(d), "contours": contours,
            "ms": {"detect": round((t1 - t0) * 1000), "segment": round((t2 - t1) * 1000)},
            "results_json": _repo_json({"detections": d})}


def detect_video(data: bytes, suffix: str, confidence: float, backend: str = "auto", max_frames: int = 300,
                 stride: int = 1):
    """Detect and track (models/tracker.ObjectTracker with config.yaml settings) frame by frame, as the GUI does."""
    import cv2
    from models.tracker import ObjectTracker
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise InferenceError("Could not open that file as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = ObjectTracker(load_config(), _system_info())
        frames, index = [], 0
        t0 = time.perf_counter()
        with _LOCK:
            det, _ = _models(backend)
            det.set_confidence(confidence)
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % stride == 0:
                    d = det.detect(frame)
                    tr = tracker.update(d)
                    active = []
                    for tid in tr["active_tracks"]:
                        boxes = tr["tracks"][tid]["boxes"]
                        trail = [[round(float(b[0] + b[2]) / 2, 1), round(float(b[1] + b[3]) / 2, 1)]
                                 for b in boxes[-20:]]
                        active.append({"id": int(tid), "box": [round(float(v), 2) for v in boxes[-1][:4]],
                                       "trail": trail})
                    frames.append({"index": index, "detections": _detections_json(d), "tracks": active})
                index += 1
        cap.release()
    finally:
        os.unlink(path)
    return {"width": w, "height": h, "fps": fps, "total_frames": total, "processed": len(frames), "stride": stride,
            "backend": type(det).__name__, "tracker": tracker.tracker_type if tracker.tracker is not None else "simple IoU",
            "confidence": confidence, "seconds": round(time.perf_counter() - t0, 1), "frames": frames,
            "n_tracks": len(tracker.tracks)}
