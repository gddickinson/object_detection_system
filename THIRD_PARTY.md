# Third-party models and libraries

This project's code is MIT licensed (LICENSE). The models it runs are not part of the repository and keep their own
licences. Anyone who downloads, uses or redistributes them must follow those terms.

| Component | Used for | How it gets here | Licence |
|---|---|---|---|
| Ultralytics YOLOv8 weights (`yolov8n.pt`) and the `ultralytics` package | Detection (`ObjectDetector`), ByteTrack tracking | ultralytics downloads `data/models/yolov8n.pt` on first use; `pip install ultralytics` | AGPL-3.0 (see https://github.com/ultralytics/ultralytics and https://ultralytics.com/license) |
| Core ML export of YOLOv8 (`yolov8n.mlpackage`) | Detection on Apple Silicon (`CoreMLDetector`) | You export it yourself (README); a derivative of the YOLOv8 weights | AGPL-3.0, as the weights |
| Segment Anything (SAM) checkpoint and `segment_anything` package | Segmentation (`ObjectSegmentor`) | Optional download (README) | Apache-2.0 (see https://github.com/facebookresearch/segment-anything) |
| TensorFlow SavedModel | `TensorFlowDetector` | Supplied by you | That model's licence |

The model files are listed in `.gitignore`. Earlier commits of this repository did include `data/models/yolov8n.pt` and
`data/models/yolov8n.mlpackage`; they were removed from the tree and purged from the git history
(history rewrite) on 2026-11-12.

The static web site (GitHub Pages) runs no models and contains no model weights.
