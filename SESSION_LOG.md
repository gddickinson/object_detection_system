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
- (Resolved later the same day, see below:) tracker threshold and committed weights.
- Tests: `pytest tests` 31 passed (torch env), `test_object_detection.py` 15 passed; node track_engine 26 cases.

## 2026-11-12 (later): tracking and licence fixes (Claude, after George's "fix any other problems")
- Fix: ByteTrack now works with ultralytics 8.x (`models/bytetrack_adapter.py`: args namespace + Boxes input). The
  simple IoU fallback has its own `simple_iou_thresh` (config.yaml, default 0.3) instead of reusing ByteTrack's
  match_thresh 0.8. On the 6-object synthetic test sequence, ByteTrack and the fallback each give 6 tracks (before: ~200).
- Fix: create_detector falls back to YOLOv8 when the Core ML model file is missing.
- Fix (licence): the Ultralytics YOLOv8 weights (AGPL-3.0) are no longer tracked (`git rm --cached`, ignored);
  ultralytics downloads yolov8n.pt on first use; Core ML is exported locally. THIRD_PARTY.md added. The old weight files
  remain in git history (removing them would need a history rewrite: George's call).
- Tests: `pytest tests` 36 passed; CI-like venv `tests/test_web.py` 18 passed, 2 skipped; `test_object_detection.py`
  15 passed; node 26 cases; screenshots --static and --base all ok, phone overflow 0 px.
