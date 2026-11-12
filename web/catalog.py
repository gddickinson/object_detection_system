"""What the repo contains, read from its own files: config.yaml, data/models/labels.txt, the model files and the Python
modules (via ast, without importing torch or Qt). Used by the home, Architecture and Models pages.
"""
import ast
from pathlib import Path

import yaml

from postprocess.colors import class_color_bgr

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [("main.py", "Entry point"), ("models", "Models"), ("postprocess", "Post-processing"), ("gui", "GUI"),
            ("utils", "Utilities"), ("auth", "Authentication"), ("plugins", "Plugins")]

# Model back ends as implemented in models/ (descriptions follow the code; see the 'source' file for each).
DETECTORS = [
    {"id": "yolov8", "name": "YOLOv8 (PyTorch)", "cls": "ObjectDetector", "source": "models/detector.py",
     "library": "ultralytics", "weights": "data/models/yolov8n.pt",
     "how": "ultralytics YOLO(model).predict with conf = the confidence threshold; boxes as xyxy pixels, class "
            "names from the model. Runs on CUDA, MPS or CPU (BaseModel picks the device from the system info).",
     "chosen": "create_detector's default, and its fallback when Core ML is unavailable."},
    {"id": "coreml", "name": "YOLOv8 (Core ML)", "cls": "CoreMLDetector", "source": "models/detector.py",
     "library": "coremltools", "weights": "data/models/yolov8n.mlpackage",
     "how": "Image resized to 640 x 640; the exported model's NMS outputs (per-class scores, normalised centre "
            "boxes) are parsed by postprocess/coreml_output.py and scaled back to the image.",
     "chosen": "create_detector on macOS / Apple Silicon when coremltools is installed (config.yaml default)."},
    {"id": "tensorflow", "name": "TensorFlow SavedModel", "cls": "TensorFlowDetector", "source": "models/detector.py",
     "library": "tensorflow + object_detection", "weights": "data/models/tf_model",
     "how": "detection_boxes / scores / classes of a TF Object Detection API SavedModel, filtered by the threshold; "
            "names from label_map.pbtxt.",
     "chosen": "Not chosen by create_detector; must be constructed directly."},
]
SEGMENTORS = [
    {"id": "sam", "name": "Segment Anything (SAM)", "cls": "ObjectSegmentor", "source": "models/segmentor.py",
     "library": "segment_anything", "weights_key": "checkpoint",
     "how": "SamPredictor with each detection box as the prompt (single mask); the largest external contour of the "
            "mask is kept as the outline."},
    {"id": "opencv", "name": "OpenCV fallback", "cls": "ObjectSegmentor._basic_segment", "source": "models/segmentor.py",
     "library": "opencv-python",
     "how": "Used when SAM or its checkpoint is missing: inside each box, grayscale, 5 x 5 Gaussian blur, adaptive "
            "Gaussian threshold (block 11, C 2, inverted) and the largest contour; the whole box if none is found."},
    {"id": "box", "name": "Box masks", "cls": "SimpleSegmentor", "source": "models/segmentor.py",
     "library": "numpy", "how": "Each detection box becomes a rectangular mask and contour."},
]
TRACKERS = [
    {"id": "bytetrack", "name": "ByteTrack", "source": "models/tracker.py",
     "how": "Imports ultralytics.trackers.byte_track.byte_tracker.BYTETracker with keyword arguments. That module "
            "path and signature do not exist in ultralytics 8.x, so creation fails and the simple IoU tracker runs "
            "instead (logged as 'ByteTrack not available')."},
    {"id": "sort", "name": "SORT", "source": "models/tracker.py",
     "how": "sort.sort.Sort(max_age = track_buffer, min_hits = 3, iou_threshold = match_thresh) if the sort package "
            "is installed; otherwise the simple IoU tracker."},
    {"id": "simple", "name": "Simple IoU tracker", "source": "postprocess/iou_tracker.py",
     "how": "Greedy: each live track takes the unmatched same-class detection with the highest IoU against its last "
            "box, if IoU > match_thresh; other detections start new tracks; a track stays live for track_buffer "
            "frames. This is what runs in practice (and what the Tracker page runs in your browser)."},
]
LLMS = [
    {"id": "ollama", "name": "Ollama (local)", "cls": "OllamaLLM", "source": "models/llm_integration.py"},
    {"id": "cloud", "name": "Cloud API", "cls": "CloudLLM", "source": "models/llm_integration.py"},
    {"id": "command", "name": "Command-line LLM", "cls": "LocalCommandLLM", "source": "models/llm_integration.py"},
]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def public_config():
    """The model and processing settings from config.yaml (no auth or API-key fields)."""
    cfg = load_config()
    llm = {k: v for k, v in (cfg.get("llm") or {}).items() if k in ("type", "model")}
    return {"models": cfg.get("models", {}), "processing": cfg.get("processing", {}), "llm": llm}


def labels():
    """COCO class names from data/models/labels.txt with the GUI's box colour for each (as #rrggbb)."""
    names = [n.strip() for n in (ROOT / "data/models/labels.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
    out = []
    for i, n in enumerate(names):
        b, g, r = class_color_bgr(n)
        out.append({"id": i, "name": n, "color": f"#{r:02x}{g:02x}{b:02x}"})
    return out


def _size(rel):
    p = ROOT / rel
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return None


def models():
    cfg = load_config().get("models", {})
    seg_ckpt = (cfg.get("segmentor") or {}).get("checkpoint", "")
    dets = [{**d, "weights_bytes": _size(d["weights"])} for d in DETECTORS]
    segs = []
    for s in SEGMENTORS:
        s = dict(s)
        if s.pop("weights_key", None):
            s["weights"] = seg_ckpt
            s["weights_bytes"] = _size(seg_ckpt) if seg_ckpt else None
        segs.append(s)
    return {"detectors": dets, "segmentors": segs, "trackers": TRACKERS, "llms": LLMS, "config": cfg}


def _module_info(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = (ast.get_docstring(tree) or "").strip().splitlines()
    classes = [{"name": n.name, "bases": [ast.unparse(b) for b in n.bases],
                "methods": [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and not m.name.startswith("_")]}
               for n in tree.body if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return {"file": str(path.relative_to(ROOT)), "lines": src.count("\n") + 1, "doc": doc[0] if doc else "",
            "classes": classes, "functions": funcs}


def modules():
    """Every Python module of the desktop app, grouped by package, with its classes and functions."""
    groups = []
    for rel, title in PACKAGES:
        p = ROOT / rel
        files = [p] if p.is_file() else sorted(f for f in p.glob("*.py") if f.name != "__init__.py")
        groups.append({"package": rel, "title": title, "modules": [_module_info(f) for f in files]})
    return groups


def summary():
    m = models()
    present = [d for d in m["detectors"] if d["weights_bytes"]]
    mods = modules()
    return {
        "n_classes": len(labels()),
        "n_detectors": len(m["detectors"]),
        "n_segmentors": len(m["segmentors"]),
        "n_trackers": len(m["trackers"]),
        "n_modules": sum(len(g["modules"]) for g in mods),
        "n_lines": sum(x["lines"] for g in mods for x in g["modules"]),
        "weights_in_repo": [{"file": d["weights"], "bytes": d["weights_bytes"]} for d in present],
        "config": public_config(),
    }
