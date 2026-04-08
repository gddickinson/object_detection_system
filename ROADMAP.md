# Object Detection and Tracking System — Roadmap

## Current State
Well-architected PyQt5 application with clear module separation: `models/` (detector, segmentor, tracker, LLM integration), `gui/` (main window, visualization, settings), `auth/` (login, user manager), `utils/` (preprocessing, visualization, system info, format converter), and `plugins/`. Supports YOLOv8, SAM segmentation, ByteTrack/SORT tracking, and Ollama LLM integration. Has `config.yaml`, `requirements.txt`, and a plugin system with `BasePlugin` interface. Production-quality structure.

## Short-term Improvements
- [ ] Add input validation in `models/detector.py` for confidence threshold ranges (0.0-1.0)
- [ ] Add error handling in `models/llm_integration.py` when Ollama service is unreachable
- [ ] Improve `auth/user_manager.py` — ensure passwords are hashed (check if using plaintext in `users.json`)
- [ ] Add a loading indicator in `gui/main_window.py` during model inference
- [ ] Add keyboard shortcuts for common actions (process, open file, save results)
- [ ] Update `data/models/labels.txt` path handling to be relative to config, not hardcoded
- [ ] Add graceful degradation when SAM weights are missing — disable segmentation, don't crash

## Feature Enhancements
- [ ] Add real-time webcam/video stream processing mode
- [ ] Add object counting and statistics dashboard (count per class, size distribution)
- [ ] Support model comparison: run multiple detectors on the same image side-by-side
- [ ] Add annotation export in COCO, YOLO, and Pascal VOC formats for training data creation
- [ ] Implement a "smart crop" feature that extracts detected objects as individual images
- [ ] Add batch processing for entire directories of images with progress tracking
- [ ] Support custom YOLOv8 model loading (user-trained weights)

## Long-term Vision
- [ ] Add model fine-tuning workflow: annotate in-app, export, train, and reload
- [ ] Build a REST API for headless inference (FastAPI with async processing)
- [ ] Add multi-camera support for surveillance-style deployments
- [ ] Implement 3D object detection for depth-camera inputs (RealSense, Azure Kinect)
- [ ] Add a plugin marketplace for community-contributed detection/processing plugins
- [ ] Package as a standalone desktop app with bundled model weights

## Technical Debt
- [ ] Add unit tests for `models/detector.py`, `models/segmentor.py`, and `models/tracker.py`
- [ ] Review `plugins/plugin_manager.py` — verify plugin sandboxing and error isolation
- [ ] Clean up `__pycache__` directories across `auth/`, `gui/`, `models/` — add to `.gitignore`
- [ ] Add type hints to `utils/preprocessing.py` and `utils/visualization.py`
- [ ] Audit `config.yaml` for any hardcoded absolute paths
- [ ] Add CI pipeline with linting (ruff) and test execution
