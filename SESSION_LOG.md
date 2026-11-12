# Session log

## 2026-11-12: web interface, static site and bug fixes (Claude, branch `web-interface`)
- Added the web interface (`web/`, `run_web.py`, port 8171) and static GitHub Pages site (`scripts/build_static.py`,
  `.github/workflows/pages.yml`). Pages: Home, Detect (local only: image/video upload, the repo's models), Tracker (JS
  port of the IoU tracker on synthetic scenes), Segment (OpenCV fallback on a synthetic scene, precomputed), Architecture,
  Models, About. No model weights or models on the static site; data/examples/cats.jpeg is used only in the local app
  (its licence is not stated).
- New `postprocess/` package (numpy only) so the web, tests and JS fixture do not need torch.
- Bugs fixed (each with a regression test, separate commits):
  - `utils/preprocessing.py` imported `pkg_resources` (absent from current setuptools): main.py could not start; now
    `importlib.metadata`.
  - `CoreMLDetector.detect` misread the (N, 80) per-class scores of the repo's yolov8n.mlpackage and returned no
    detections (output/cats.json was empty); now parsed by `postprocess/coreml_output.py`.
  - `create_detector` fallback kept type 'coreml' when Core ML was unavailable, so ObjectDetector loaded nothing; now
    falls back to 'yolov8'.
  - Class colours used salted `hash()` and changed every run; now crc32 (`postprocess/colors.py`).
  - README claimed ByteTrack/SORT tracking; with ultralytics 8.x the ByteTrack import fails and the simple IoU tracker
    always runs. Docs corrected.
- Refactor (no behaviour change): `ObjectTracker._update_simple`/`_calculate_iou` now call `postprocess/iou_tracker.py`
  (checked identical on 300 random sequences).
- Open for George: the IoU tracker reads `match_thresh: 0.8` as a minimum IoU (ByteTrack meant a cost threshold), which
  breaks tracks almost every frame on moving objects (synthetic demo: 203 tracks for 6 objects at 0.8, 6 at 0.3);
  whether to repair the ByteTrack integration; `data/models/yolov8n.pt` (ultralytics weights) is committed in an MIT repo.
- Tests: `pytest tests` 31 passed (torch env), `test_object_detection.py` 15 passed; node track_engine 26 cases.
