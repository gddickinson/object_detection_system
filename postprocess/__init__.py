"""Pure post-processing used by the models and the GUI (numpy only: no torch, Qt or OpenCV).

Contents:
- coreml_output.py  parse_coreml_detections: Core ML YOLO (NMS pipeline) outputs -> boxes, scores, classes
- colors.py         class_color_bgr: stable per-class box colour (crc32 of the name -> HSV hue)
"""
