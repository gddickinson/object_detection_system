# Object Detection and Tracking System

A comprehensive computer vision application for detecting, segmenting, and tracking objects in images and videos, with optional LLM integration for enhanced object recognition.

## Features

- **Object Detection**: Identify objects in images and videos using state-of-the-art models (YOLOv8, TensorFlow, CoreML)
- **Instance Segmentation**: Generate precise object outlines using Segment Anything Model (SAM)
- **Object Tracking**: Track objects across video frames with ByteTrack or SORT algorithms
- **LLM Integration**: Enhanced object recognition using Ollama (local) or Cloud APIs
- **Multi-Format Support**: Process various image and video formats
- **Hardware Optimization**: Automatic detection and utilization of available hardware (CUDA, MPS on Apple Silicon)
- **User Authentication**: Secure login system with user management
- **Plugin System**: Extensible architecture through plugins
- **Rich Visualization**: Interactive display with zooming, panning, and object highlighting

## System Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- OpenCV 4.7 or higher
- PyQt5 5.15 or higher
- 4GB RAM minimum (8GB+ recommended)
- NVIDIA GPU with CUDA support (optional, for acceleration)
- Apple Silicon (M1/M2/M3) for MPS acceleration (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-system.git
   cd object-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download models:
   
   For YOLOv8:
   ```bash
   mkdir -p data/models
   # Download YOLOv8n (small)
   wget -O data/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```
   
   For SAM (Segment Anything Model):
   ```bash
   # Download SAM ViT-B
   wget -O data/models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

## Usage

### Starting the Application

Run the main application:

```bash
python main.py
```

With options:
```bash
# Skip login screen (development only)
python main.py --skip-login

# Specify input file
python main.py --input /path/to/image.jpg

# Debug mode
python main.py --debug
```

### Login

- Default admin credentials:
  - Username: `admin`
  - Password: `admin`
- After first login, navigate to User Management to change the admin password and/or create new users.

### Basic Workflow

1. Select an input image or video using the "Select Input" button or File → Open Input
2. Choose detection model and adjust confidence threshold
3. Enable/disable LLM integration (Ollama, Cloud API)
4. Click "Process" to analyze the input
5. View results in the visualization area
6. Save results using File → Save Results

### Ollama Integration

To use Ollama for LLM analysis:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama3 model (or another model of your choice):
   ```bash
   ollama pull llama3
   ```
3. Start the Ollama service:
   ```bash
   ollama serve
   ```
4. In the application, select "Ollama" in the LLM dropdown

## Configuration

The application settings can be configured through the GUI (Edit → Settings) or by editing the `config.yaml` file.

Key configuration options:

- **Models**: Select and configure detection, segmentation, and tracking models
- **LLM Integration**: Configure Ollama, Cloud API, or command-line LLM integration
- **System**: Configure hardware acceleration, memory usage, and logging
- **GUI**: Adjust window size, theme, and other interface settings
- **Processing**: Configure batch size, output directory, and other processing options

## Project Structure

```
object_detection_system/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── config.yaml                  # Configuration
├── README.md                    # Documentation
├── auth/                        # Authentication
│   ├── login.py                 # Login window
│   └── user_manager.py          # User authentication
├── models/                      # AI Models
│   ├── base.py                  # Base classes
│   ├── detector.py              # Object detection
│   ├── segmentor.py             # Instance segmentation
│   ├── tracker.py               # Object tracking
│   └── llm_integration.py       # LLM integration
├── utils/                       # Utilities
│   ├── preprocessing.py         # Preprocessing
│   ├── visualization.py         # Visualization
│   ├── system_info.py           # System detection
│   └── format_converter.py      # Format handling
├── gui/                         # User Interface
│   ├── main_window.py           # Main window
│   ├── visualization.py         # Visualization
│   └── settings_dialog.py       # Settings
├── plugins/                     # Plugin system
│   └── plugin_manager.py        # Plugin manager
└── data/                        # Data files
    ├── models/                  # Model weights
    ├── users/                   # User data
    └── examples/                # Example files
```

## Adding Plugins

The application supports plugins for extending functionality. To create a plugin:

1. Create a new Python file in the `plugins` directory
2. Implement the `BasePlugin` interface
3. The plugin will be automatically loaded when the application starts

Example plugin:

```python
from models.base import BasePlugin

class MyPlugin(BasePlugin):
    def get_info(self):
        return {
            'id': 'my_plugin',
            'name': 'My Plugin',
            'description': 'This is an example plugin',
            'version': '1.0',
            'author': 'Your Name',
            'type': 'processor'
        }
    
    def process(self, data):
        # Process data
        return processed_data
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for SAM
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for object tracking
- [Ollama](https://ollama.ai) for local LLM capabilities
