"""FastAPI web app for the Object Detection and Tracking System: serves the front end in static/ and the JSON API.

Run with `python run_web.py` (http://127.0.0.1:8171, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. Inference
(POST /api/detect, /api/detect_video) uses the repo's own model classes and runs only in the local app.
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, demos

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Object Detection and Tracking System"
EXAMPLE = catalog.ROOT / "data" / "examples" / "cats.jpeg"
MAX_IMAGE = 60 * 1024 * 1024
MAX_VIDEO = 800 * 1024 * 1024
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Counts and config.yaml settings for the home page."""
    return {"title": TITLE, "version": "1", **catalog.summary()}


@app.get("/api/models")
def models():
    """Detector, segmentor, tracker and LLM back ends as implemented in models/, with model-file sizes."""
    return catalog.models()


@app.get("/api/modules")
def modules():
    """Every Python module of the desktop app with its classes and functions (read with ast)."""
    return catalog.modules()


@app.get("/api/labels")
def labels():
    """The 80 class names of data/models/labels.txt with the GUI's box colour for each."""
    return catalog.labels()


@app.get("/api/demo/tracker")
def demo_tracker():
    """A synthetic detection sequence tracked by postprocess/iou_tracker.py with several IoU thresholds."""
    return demos.tracker_demo()


@app.get("/api/demo/segment")
def demo_segment():
    """OpenCV-fallback segmentation of a synthetic scene (precomputed by scripts/make_segment_demo.py)."""
    return demos.segment_demo()


@app.get("/api/status")
def status():
    """Local app only: installed packages and model files for inference."""
    from .inference import status as st
    return st()


@app.get("/api/example.jpg", include_in_schema=False)
def example_image():
    """Local app only: data/examples/cats.jpeg (not published on the static site: its licence is not stated)."""
    return FileResponse(EXAMPLE, media_type="image/jpeg")


def _conf(value: str) -> float:
    try:
        c = float(value)
    except ValueError:
        raise HTTPException(400, "confidence must be a number")
    if not 0.0 <= c <= 1.0:
        raise HTTPException(400, "confidence must be between 0 and 1")
    return c


@app.post("/api/detect")
async def detect(file: UploadFile | None = File(None), example: str = Form(""), confidence: str = Form("0.5"),
                 segment: str = Form("1"), backend: str = Form("auto")):
    """Detect (and optionally segment) objects in an uploaded image or the repo's example image."""
    from .inference import BACKENDS, InferenceError, detect_image
    if backend not in BACKENDS:
        raise HTTPException(400, "unknown back end")
    if file is not None and file.filename:
        data = await file.read(MAX_IMAGE + 1)
        if len(data) > MAX_IMAGE:
            raise HTTPException(400, "Image too large (limit 60 MB).")
    elif example == "cats":
        data = EXAMPLE.read_bytes()
    else:
        raise HTTPException(400, "Choose an image file or the example image.")
    try:
        return detect_image(data, _conf(confidence), segment == "1", backend)
    except InferenceError as e:
        raise HTTPException(503, str(e))


@app.post("/api/detect_video")
async def detect_video(file: UploadFile = File(...), confidence: str = Form("0.5"), backend: str = Form("auto"),
                       max_frames: str = Form("300"), stride: str = Form("1")):
    """Detect and track objects frame by frame in an uploaded video."""
    from .inference import BACKENDS, InferenceError, detect_video as run
    name = (file.filename or "").lower()
    if not name.endswith(VIDEO_EXT):
        raise HTTPException(400, "Unsupported video type: " + ", ".join(VIDEO_EXT))
    if backend not in BACKENDS:
        raise HTTPException(400, "unknown back end")
    try:
        n, s = max(1, min(int(max_frames), 3000)), max(1, min(int(stride), 30))
    except ValueError:
        raise HTTPException(400, "max_frames and stride must be whole numbers")
    data = await file.read(MAX_VIDEO + 1)
    if len(data) > MAX_VIDEO:
        raise HTTPException(400, "Video too large (limit 800 MB).")
    try:
        return run(data, Path(name).suffix, _conf(confidence), backend, n, s)
    except InferenceError as e:
        raise HTTPException(503, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/models", "/modules", "/labels", "/demo/tracker", "/demo/segment"]
