from abc import ABC, abstractmethod
import logging

logger = logging.getLogger('object_detection.models.base')

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config=None, system_info=None):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        self.config = config or {}
        self.system_info = system_info or {}
        self.device = self._determine_device()
    
    def _determine_device(self):
        """
        Determine the device to use for computation.
        
        Returns:
            Device string ('cuda', 'mps', 'cpu')
        """
        # Get acceleration type from system info
        acceleration = self.system_info.get('acceleration', 'CPU')
        
        if acceleration == 'CUDA':
            return 'cuda'
        elif acceleration == 'MPS':
            return 'mps'
        else:
            return 'cpu'


class BaseDetector(BaseModel):
    """Abstract base class for object detectors."""
    
    @abstractmethod
    def detect(self, image):
        """
        Detect objects in an image.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary containing detection results
        """
        pass
    
    @abstractmethod
    def set_confidence(self, confidence):
        """
        Set the confidence threshold.
        
        Args:
            confidence: Confidence threshold value
        """
        pass


class BaseSegmentor(BaseModel):
    """Abstract base class for object segmentation models."""
    
    @abstractmethod
    def segment(self, image, boxes):
        """
        Segment objects in an image based on bounding boxes.
        
        Args:
            image: Input image
            boxes: Bounding boxes
        
        Returns:
            Dictionary containing segmentation results
        """
        pass


class BaseTracker(BaseModel):
    """Abstract base class for object trackers."""
    
    @abstractmethod
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: Detection results
        
        Returns:
            Dictionary containing tracking results
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the tracker state."""
        pass


class BasePlugin(ABC):
    """Abstract base class for plugins."""
    
    def __init__(self, config=None):
        """
        Initialize the plugin.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def get_info(self):
        """
        Get plugin information.
        
        Returns:
            Dictionary containing plugin information
        """
        pass
    
    @abstractmethod
    def process(self, data):
        """
        Process data.
        
        Args:
            data: Input data
        
        Returns:
            Processed data
        """
        pass
