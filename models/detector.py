import torch
import numpy as np
import logging
import cv2
import os
from .base import BaseDetector

logger = logging.getLogger('object_detection.models.detector')

class ObjectDetector(BaseDetector):
    """Class for object detection using YOLO models."""

    def __init__(self, config=None, system_info=None):
        """
        Initialize the object detector.

        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)

        # Get detector settings from config
        detector_config = self.config.get('models', {}).get('detector', {})

        self.model_type = detector_config.get('type', 'yolov8')
        self.model_name = detector_config.get('model_name', 'yolov8n.pt')
        self.confidence = detector_config.get('confidence', 0.5)

        # Initialize the model
        self.model = self._init_model()

    def _init_model(self):
        """
        Initialize the detection model.

        Returns:
            Initialized model
        """
        try:
            if self.model_type == 'yolov8':
                # Use ultralytics YOLO
                from ultralytics import YOLO

                # Check for model file
                model_path = self.model_name
                if not os.path.exists(model_path):
                    # Try looking in the data/models directory
                    model_path = os.path.join('data', 'models', self.model_name)

                # Load model with appropriate device
                model = YOLO(model_path)

                # Set device
                if self.device == 'cuda' and torch.cuda.is_available():
                    logger.info(f"Using CUDA for YOLOv8")
                    model.to('cuda')
                elif self.device == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    logger.info(f"Using MPS for YOLOv8")
                    model.to('mps')
                else:
                    logger.info(f"Using CPU for YOLOv8")
                    model.to('cpu')

                return model
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None

        except Exception as e:
            logger.error(f"Error initializing detection model: {e}")
            return None

    def detect(self, image):
        """
        Detect objects in an image.

        Args:
            image: Input image (numpy array)

        Returns:
            Dictionary containing:
                - boxes: Bounding boxes (x1, y1, x2, y2)
                - scores: Confidence scores
                - classes: Class IDs
                - class_names: Class names
                - centers: Center points (x, y) of each box
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

        try:
            # Clone image to avoid modifying the original
            if isinstance(image, np.ndarray):
                img = image.copy()
            else:
                img = image

            # Run inference
            results = self.model(img, conf=self.confidence)

            boxes = []
            scores = []
            classes = []
            class_names = []
            centers = []

            for result in results:
                # Get boxes
                if len(result.boxes) > 0:
                    boxes.extend(result.boxes.xyxy.cpu().numpy())
                    scores.extend(result.boxes.conf.cpu().numpy())

                    # Convert class IDs to integers
                    detected_classes = result.boxes.cls.cpu().numpy().astype(int)
                    classes.extend(detected_classes)

                    # Get class names
                    for cls_id in detected_classes:
                        class_names.append(self.model.names[cls_id])

                    # Calculate centers
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        centers.append((center_x, center_y))

            return {
                'boxes': np.array(boxes),
                'scores': np.array(scores),
                'classes': np.array(classes),
                'class_names': class_names,
                'centers': np.array(centers)
            }

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

    def set_confidence(self, confidence):
        """
        Set the confidence threshold.

        Args:
            confidence: Confidence threshold value (must be between 0.0 and 1.0)

        Raises:
            ValueError: If confidence is not in the range [0.0, 1.0]
        """
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        self.confidence = float(confidence)


class TensorFlowDetector(BaseDetector):
    """Class for object detection using TensorFlow models."""

    def __init__(self, config=None, system_info=None):
        """
        Initialize the TensorFlow detector.

        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)

        # Get detector settings from config
        detector_config = self.config.get('models', {}).get('detector', {})

        self.model_path = detector_config.get('model_path', 'data/models/tf_model')
        self.confidence = detector_config.get('confidence', 0.5)

        # Initialize the model
        self.model = self._init_model()
        self.category_index = self._load_labels()

    def _init_model(self):
        """
        Initialize the TensorFlow model.

        Returns:
            Initialized model
        """
        try:
            import tensorflow as tf

            # Check if model directory exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path not found: {self.model_path}")
                return None

            # Load saved model
            model = tf.saved_model.load(self.model_path)

            return model

        except Exception as e:
            logger.error(f"Error initializing TensorFlow model: {e}")
            return None

    def _load_labels(self):
        """
        Load label map.

        Returns:
            Category index
        """
        try:
            import tensorflow as tf
            from object_detection.utils import label_map_util

            # Look for label map in model directory
            label_map_path = os.path.join(self.model_path, 'label_map.pbtxt')

            if not os.path.exists(label_map_path):
                # Try default location
                label_map_path = 'data/models/label_map.pbtxt'

            if not os.path.exists(label_map_path):
                logger.error(f"Label map not found: {label_map_path}")
                return {}

            category_index = label_map_util.create_category_index_from_labelmap(
                label_map_path, use_display_name=True)

            return category_index

        except Exception as e:
            logger.error(f"Error loading label map: {e}")
            return {}

    def detect(self, image):
        """
        Detect objects in an image.

        Args:
            image: Input image (numpy array)

        Returns:
            Dictionary containing:
                - boxes: Bounding boxes (x1, y1, x2, y2)
                - scores: Confidence scores
                - classes: Class IDs
                - class_names: Class names
                - centers: Center points (x, y) of each box
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

        try:
            import tensorflow as tf

            # Clone image to avoid modifying the original
            img = image.copy()

            # Convert image to tensor
            input_tensor = tf.convert_to_tensor(img)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run inference
            detections = self.model(input_tensor)

            # Process detections
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)

            # Filter by confidence
            indices = scores >= self.confidence
            boxes = boxes[indices]
            scores = scores[indices]
            classes = classes[indices]

            # Convert boxes from [y1, x1, y2, x2] to [x1, y1, x2, y2]
            height, width = img.shape[:2]
            boxes_xyxy = []
            centers = []
            class_names = []

            for box, cls in zip(boxes, classes):
                y1, x1, y2, x2 = box
                # Convert to absolute coordinates
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)

                boxes_xyxy.append([x1, y1, x2, y2])

                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y))

                # Get class name
                if cls in self.category_index:
                    class_names.append(self.category_index[cls]['name'])
                else:
                    class_names.append(f"class_{cls}")

            return {
                'boxes': np.array(boxes_xyxy),
                'scores': scores,
                'classes': classes,
                'class_names': class_names,
                'centers': np.array(centers)
            }

        except Exception as e:
            logger.error(f"Error detecting objects with TensorFlow: {e}")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

    def set_confidence(self, confidence):
        """
        Set the confidence threshold.

        Args:
            confidence: Confidence threshold value (must be between 0.0 and 1.0)

        Raises:
            ValueError: If confidence is not in the range [0.0, 1.0]
        """
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        self.confidence = float(confidence)


class CoreMLDetector(BaseDetector):
    """Class for object detection using Core ML models on macOS."""

    def __init__(self, config=None, system_info=None):
        """
        Initialize the Core ML detector.

        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)

        # Get detector settings from config
        detector_config = self.config.get('models', {}).get('detector', {})

        self.model_path = detector_config.get('coreml_model_path', 'data/models/yolov8.mlpackage')
        self.confidence = detector_config.get('confidence', 0.5)

        # Initialize the model
        self.model = self._init_model()
        self.labels = self._load_labels()

    def _init_model(self):
        """
        Initialize the Core ML model.

        Returns:
            Initialized model
        """
        try:
            import coremltools as ct

            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path not found: {self.model_path}")
                return None

            # Load model
            model = ct.models.MLModel(self.model_path)

            return model

        except ImportError:
            logger.warning("CoreML Tools not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing Core ML model: {e}")
            return None

    def _load_labels(self):
        """
        Load labels.

        Returns:
            Dictionary of labels
        """
        try:
            # Look for labels in model directory
            labels_path = os.path.join(os.path.dirname(self.model_path), 'labels.txt')

            if not os.path.exists(labels_path):
                # Try default location
                labels_path = 'data/models/labels.txt'

            if not os.path.exists(labels_path):
                logger.warning(f"Labels file not found: {labels_path}")
                return {}

            labels = {}
            with open(labels_path, 'r') as f:
                for i, line in enumerate(f):
                    labels[i] = line.strip()

            return labels

        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return {}

    def detect(self, image):
        """
        Detect objects in an image.

        Args:
            image: Input image (numpy array)

        Returns:
            Dictionary containing:
                - boxes: Bounding boxes (x1, y1, x2, y2)
                - scores: Confidence scores
                - classes: Class IDs
                - class_names: Class names
                - centers: Center points (x, y) of each box
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

        try:
            from PIL import Image

            # Get original image dimensions
            original_height, original_width = image.shape[:2]

            # Convert numpy array to PIL Image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Resize to the expected input size (640x640 for YOLOv8)
            input_size = (640, 640)
            resized_img = img.resize(input_size)

            # Run inference
            prediction = self.model.predict({"image": resized_img})

            # Process results (this will vary depending on the CoreML model's output format)
            boxes = []
            scores = []
            classes = []
            class_names = []
            centers = []

            # Extract detection results based on model output
            # Format may vary based on the specific CoreML conversion
            if "coordinates" in prediction and "confidence" in prediction:
                coords = prediction["coordinates"]
                confs = prediction["confidence"]
                class_ids = prediction.get("classId", [0] * len(coords))

                # Scale boxes back to original image size
                scale_x = original_width / input_size[0]
                scale_y = original_height / input_size[1]

                for i, box in enumerate(coords):
                    if confs[i] >= self.confidence:
                        # Extract box coordinates (may need to be adjusted based on model output format)
                        x1, y1, x2, y2 = box

                        # Scale to original image
                        x1 = x1 * scale_x
                        y1 = y1 * scale_y
                        x2 = x2 * scale_x
                        y2 = y2 * scale_y

                        boxes.append([x1, y1, x2, y2])
                        scores.append(confs[i])

                        # Get class ID and name
                        class_id = class_ids[i] if isinstance(class_ids, list) else class_ids
                        classes.append(class_id)

                        # Get class name
                        if class_id in self.labels:
                            class_names.append(self.labels[class_id])
                        else:
                            class_names.append(f"class_{class_id}")

                        # Calculate center
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        centers.append((center_x, center_y))

            # Alternative format (depends on CoreML model output)
            elif "output" in prediction:
                # This format is used by some YOLOv8 conversions to CoreML
                detections = prediction["output"]

                # Process based on the expected format
                # The exact processing depends on the model output structure
                # This is a simplified example
                for detection in detections:
                    if len(detection) >= 6 and detection[4] >= self.confidence:
                        class_id = int(detection[5])
                        confidence = detection[4]

                        # Box coordinates (need to adapt based on model output)
                        x1, y1, x2, y2 = detection[0:4]

                        # Scale to original image
                        scale_x = original_width / input_size[0]
                        scale_y = original_height / input_size[1]

                        x1 = x1 * scale_x
                        y1 = y1 * scale_y
                        x2 = x2 * scale_x
                        y2 = y2 * scale_y

                        boxes.append([x1, y1, x2, y2])
                        scores.append(confidence)
                        classes.append(class_id)

                        # Get class name
                        if class_id in self.labels:
                            class_names.append(self.labels[class_id])
                        else:
                            class_names.append(f"class_{class_id}")

                        # Calculate center
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        centers.append((center_x, center_y))

            # Fallback to using PyTorch model if CoreML fails
            if not boxes and hasattr(self, 'pt_model') and self.pt_model is not None:
                logger.warning("CoreML detection failed, falling back to PyTorch")
                return self.pt_model.detect(image)

            return {
                'boxes': np.array(boxes),
                'scores': np.array(scores),
                'classes': np.array(classes),
                'class_names': class_names,
                'centers': np.array(centers)
            }

        except Exception as e:
            logger.error(f"Error detecting objects with Core ML: {e}")

            # If possible, fall back to PyTorch model
            if hasattr(self, 'pt_model') and self.pt_model is not None:
                logger.warning("CoreML detection failed, falling back to PyTorch")
                return self.pt_model.detect(image)

            # Otherwise, return empty results
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'class_names': [],
                'centers': np.array([])
            }

    def set_confidence(self, confidence):
        """
        Set the confidence threshold.

        Args:
            confidence: Confidence threshold value (must be between 0.0 and 1.0)

        Raises:
            ValueError: If confidence is not in the range [0.0, 1.0]
        """
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        self.confidence = float(confidence)


def create_detector(config=None, system_info=None):
    """
    Factory function to create an appropriate detector for the current system.

    Args:
        config: Configuration dictionary
        system_info: System information dictionary

    Returns:
        Detector instance
    """
    if not system_info:
        system_info = {}

    # Get OS and acceleration type
    os_type = system_info.get('os', '')
    acceleration = system_info.get('acceleration', 'CPU')

    # For macOS with Apple Silicon, try Core ML first
    if os_type == 'Darwin' and 'arm' in system_info.get('architecture', ''):
        try:
            import coremltools
            logger.info("Using Core ML detector for Apple Silicon")
            return CoreMLDetector(config, system_info)
        except ImportError:
            logger.info("Core ML not available, falling back to standard detector")

    # Default to YOLO
    return ObjectDetector(config, system_info)
