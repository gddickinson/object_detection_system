# Object Detection and Tracking System - Interface Map

## Module Overview

| Directory/File | Purpose |
|---------------|---------|
| `main.py` | Entry point: argument parsing, config loading, launches PyQt5 GUI |
| `config.yaml` | All configuration: models, auth, GUI, processing, plugins |
| `models/` | Detection, segmentation, tracking, and LLM integration models |
| `gui/` | PyQt5 GUI: main window, settings dialog, visualization widget |
| `auth/` | Authentication: login window, user management with salted hashing |
| `utils/` | Preprocessing, visualization, system info, format converter |
| `plugins/` | Plugin system with BasePlugin interface and PluginManager |
| `postprocess/` | Pure numpy post-processing shared by models, GUI and web: Core ML output parsing, class colours, IoU tracker |
| `web/`, `run_web.py` | Web interface (FastAPI, port 8171) and static-site data; see "Web interface" below |
| `scripts/` | Static build, screenshots, fixture/demo generators |
| `tests/` | pytest: post-processing, bug regressions, web API (`tests/fixtures/` JS-vs-Python cases) |
| `data/` | Labels, example image, user data; model weights are downloaded or exported locally (git-ignored, THIRD_PARTY.md) |
| `test_object_detection.py` | Test suite for models, auth, utils, config |

## Key Classes and Functions

### models/base.py
- `BaseModel` - abstract base with device detection (cpu/cuda/mps)
- `BaseDetector` - abstract: `detect(image)`, `set_confidence(c)`
- `BaseSegmentor` - abstract: `segment(image, boxes)`
- `BaseTracker` - abstract: `update(detections)`, `reset()`
- `BasePlugin` - abstract: `get_info()`, `process(data)`

### models/detector.py
- `ObjectDetector` - YOLOv8 via ultralytics (confidence validated 0.0-1.0)
- `TensorFlowDetector` - TensorFlow SavedModel detector
- `CoreMLDetector` - Apple CoreML detector for Apple Silicon
- `create_detector(config, system_info)` - factory: Core ML on macOS arm64 with coremltools, else YOLOv8 (a 'coreml'
  config falls back to type 'yolov8')

### models/segmentor.py
- `ObjectSegmentor` - SAM-based segmentation with OpenCV fallback
- `SimpleSegmentor` - basic box-based segmentation

### models/tracker.py
- `ObjectTracker` - ByteTrack (models/bytetrack_adapter.py, ultralytics) or SORT; else the simple IoU tracker (postprocess/iou_tracker.py, `simple_iou_thresh`)

### models/llm_integration.py
- `BaseLLM` - abstract: `analyze_image(image)`
- `CloudLLM` - OpenAI/Anthropic API integration
- `OllamaLLM` - local Ollama with connection error handling
- `LocalCommandLLM` - command-line LLM wrapper
- `LLMAnalyzer` - facade that selects LLM backend from config

### auth/user_manager.py
- `UserManager` - CRUD for users with SHA-256 salted password hashing
- `UserManagementDialog` / `AddUserDialog` / `EditUserDialog` - PyQt5 dialogs

### utils/preprocessing.py
- `check_dependencies(system_info)` - verifies required/optional packages
- `is_package_installed(name)` - single package check
- `check_gpu_availability(system_info)` - CUDA/MPS detection
- `check_ollama_availability()` - Ollama service check
- `download_model_if_needed(path, url)` - model downloader with progress

### plugins/plugin_manager.py
- `PluginManager` - discovers, loads, and manages plugins from directory

## Data Flow
```
main.py -> config.yaml (load settings)
        -> auth/login.py (authenticate)
        -> gui/main_window.py (launch UI)
           -> models/detector.py (detect objects)
           -> models/segmentor.py (segment objects)
           -> models/tracker.py (track across frames)
           -> models/llm_integration.py (LLM descriptions)
           -> gui/visualization.py (render results)
```

### postprocess/ (numpy only; no torch, Qt or OpenCV)
- `coreml_output.py` - `parse_coreml_detections(coordinates, confidence, w, h, threshold)`: Core ML NMS outputs -> pixel boxes
- `colors.py` - `class_color_bgr(name)`: stable per-class colour (crc32 hue), used by gui/ and utils/visualization.py
- `iou_tracker.py` - `box_iou`, `associate_simple(...)`: the greedy IoU tracker used by ObjectTracker

### models/bytetrack_adapter.py
- `create_bytetrack(track_thresh, track_buffer, match_thresh)`, `bytetrack_update(tracker, boxes, scores, classes)` - ultralytics 8.x BYTETracker with numpy input

## Web interface (`web/` package, `python run_web.py`, http://127.0.0.1:8171)
| File | Contents |
|------|----------|
| `web/app.py` | FastAPI: `/api/info`, `/models`, `/modules`, `/labels`, `/demo/tracker`, `/demo/segment` (all in `STATIC_ENDPOINTS`); local only: `/status`, `/example.jpg`, `POST /detect`, `POST /detect_video` |
| `web/catalog.py` | Back-end descriptions, `public_config` (no auth/API-key fields), `labels`, `modules` (ast scan), `summary` |
| `web/synthetic.py` | `make_sequence` (synthetic detections with ground truth), `run_tracker` (associate_simple + identity changes) |
| `web/demos.py` | `tracker_demo` (config.yaml settings + other thresholds), `segment_demo` (reads `web/demo/segment_demo.json`) |
| `web/inference.py` | Lazy model creation with the repo's classes, `detect_image`, `detect_video` (ObjectTracker), `status` |
| `scripts/build_static.py` | Static export (every STATIC_ENDPOINTS GET to JSON) |
| `scripts/screenshots.py` | Playwright smoke test + `docs/img/*.png` |
| `scripts/make_tracker_fixture.py` | Writes `tests/fixtures/tracker_cases.json` (IoU, colours, tracker runs) |
| `scripts/make_segment_demo.py` | Synthetic scene + ObjectSegmentor OpenCV fallback -> `web/static/demo/segment_scene.png`, `web/demo/segment_demo.json` (needs torch) |

### Front end (`web/static/`, ES modules, no build step)
| File | Contents |
|------|----------|
| `index.html`, `style.css` (kit theme), `style_app.css`, `icon.svg` | Shell and styles |
| `core.js` | Kit core: `api`, router, `slider`, `needsServer`, `animate`, `download`, `rng` |
| `main.js` | Lazy page registration, mode label, drifting-boxes background |
| `static_api.js` | Static site: exported JSON; POSTs need the local app |
| `track_engine.js` | JS port: `boxIou`, `associateSimple`, `runTracker`, `classColorBgr`/`classColor` (crc32); tested by `track_engine.test.mjs` |
| `scene.js` | Synthetic sequences in the browser (`makeSequence`), `backdrop` |
| `draw.js` | Overlay drawing after gui/visualization.py: `drawDetections`, `drawSegments`, `drawTracks` |
| `home.js`, `detect.js`, `tracker.js`, `segment.js`, `arch.js`, `models.js`, `about.js` | Pages |
