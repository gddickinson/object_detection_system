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
| `data/` | Model weights, labels, example images, user data |
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
- `create_detector(config, system_info)` - factory function

### models/segmentor.py
- `ObjectSegmentor` - SAM-based segmentation with OpenCV fallback
- `SimpleSegmentor` - basic box-based segmentation

### models/tracker.py
- Object tracking (ByteTrack/SORT)

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
