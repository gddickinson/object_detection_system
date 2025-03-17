import torch
import numpy as np
import cv2
import os
import logging
from .base import BaseSegmentor

logger = logging.getLogger('object_detection.models.segmentor')

class ObjectSegmentor(BaseSegmentor):
    """Class for object segmentation using Segment Anything Model (SAM)."""
    
    def __init__(self, config=None, system_info=None):
        """
        Initialize the object segmentor.
        
        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)
        
        # Get segmentor settings from config
        segmentor_config = self.config.get('models', {}).get('segmentor', {})
        
        self.model_type = segmentor_config.get('type', 'sam')
        self.sam_model_type = segmentor_config.get('model_type', 'vit_h')
        self.checkpoint_path = segmentor_config.get('checkpoint', 'sam_vit_h_4b8939.pth')
        
        # Initialize model if available
        self.sam = None
        self.predictor = None
        
        try:
            # Only initialize if SAM is installed
            self._init_model()
        except ImportError:
            logger.warning("SAM not available, using basic segmentation")
        except Exception as e:
            logger.error(f"Error initializing SAM: {e}")
    
    def _init_model(self):
        """Initialize the segmentation model."""
        try:
            # Import SAM
            from segment_anything import sam_model_registry, SamPredictor
            
            # Check for checkpoint file
            model_path = self.checkpoint_path
            if not os.path.exists(model_path):
                # Try looking in the data/models directory
                model_path = os.path.join('data', 'models', self.checkpoint_path)
            
            if not os.path.exists(model_path):
                logger.warning(f"SAM checkpoint not found: {model_path}")
                logger.warning("Using basic segmentation instead")
                return
            
            # Initialize SAM
            self.sam = sam_model_registry[self.sam_model_type](checkpoint=model_path)
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                logger.info("Using CUDA for SAM")
                self.sam.to('cuda')
            elif self.device == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
                logger.info("Using MPS for SAM")
                self.sam.to('mps')
            else:
                logger.info("Using CPU for SAM")
                self.sam.to('cpu')
            
            # Create predictor
            self.predictor = SamPredictor(self.sam)
            
            logger.info(f"Initialized SAM with {self.sam_model_type} model")
            
        except ImportError:
            logger.warning("segment_anything package not found")
            raise
        except Exception as e:
            logger.error(f"Error initializing SAM: {e}")
            raise
    
    def segment(self, image, boxes):
        """
        Segment objects in an image based on bounding boxes.
        
        Args:
            image: Input image (numpy array)
            boxes: Bounding boxes (N, 4) where each box is (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing:
                - masks: Binary masks for each object
                - contours: Contour points for each object
        """
        if len(boxes) == 0:
            return {'masks': [], 'contours': []}
        
        # If SAM is not available, use basic segmentation
        if self.sam is None or self.predictor is None:
            return self._basic_segment(image, boxes)
        
        try:
            # Set image for predictor
            self.predictor.set_image(image)
            
            masks = []
            contours = []
            
            # Process each box
            for box in boxes:
                # Convert box to the format expected by SAM
                x1, y1, x2, y2 = map(float, box)
                
                # Ensure box coordinates are valid
                height, width = image.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    masks.append(np.zeros((height, width), dtype=bool))
                    contours.append(None)
                    continue
                
                # Convert box to tensor and format for SAM
                box_tensor = torch.tensor([[x1, y1, x2, y2]], device=self.predictor.device)
                
                # Generate mask for this box
                masks_pred, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=box_tensor,
                    multimask_output=False,
                )
                
                # Convert mask to numpy
                mask = masks_pred[0, 0].cpu().numpy()
                masks.append(mask)
                
                # Extract contour
                contour = self._extract_contour(mask)
                contours.append(contour)
            
            return {
                'masks': masks,
                'contours': contours
            }
            
        except Exception as e:
            logger.error(f"Error in SAM segmentation: {e}")
            # Fallback to basic segmentation
            return self._basic_segment(image, boxes)
    
    def _basic_segment(self, image, boxes):
        """
        Basic segmentation using OpenCV.
        
        Args:
            image: Input image
            boxes: Bounding boxes
            
        Returns:
            Dictionary with masks and contours
        """
        height, width = image.shape[:2]
        masks = []
        contours = []
        
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box coordinates are valid
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                masks.append(np.zeros((height, width), dtype=bool))
                contours.append(None)
                continue
            
            # Extract object region
            roi = image[y1:y2, x1:x2]
            
            # Skip if ROI is empty
            if roi.size == 0:
                masks.append(np.zeros((height, width), dtype=bool))
                contours.append(None)
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours in the thresholded image
            roi_contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create mask for the full image
            mask = np.zeros((height, width), dtype=bool)
            
            # If contours were found, use the largest one
            if roi_contours:
                # Find the largest contour
                largest_contour = max(roi_contours, key=cv2.contourArea)
                
                # Create a temporary mask for the ROI
                roi_mask = np.zeros_like(thresh, dtype=np.uint8)
                cv2.drawContours(roi_mask, [largest_contour], 0, 255, -1)
                
                # Transfer the ROI mask to the full image mask
                mask[y1:y2, x1:x2] = roi_mask > 0
                
                # Adjust contour coordinates to the full image
                full_contour = largest_contour.copy()
                full_contour[:, :, 0] += x1
                full_contour[:, :, 1] += y1
                
                contours.append(full_contour)
            else:
                # If no contours found, use the entire box
                mask[y1:y2, x1:x2] = True
                
                # Create a rectangular contour
                rect_contour = np.array([
                    [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                ], dtype=np.int32)
                
                contours.append(rect_contour)
            
            masks.append(mask)
        
        return {
            'masks': masks,
            'contours': contours
        }
    
    def _extract_contour(self, mask):
        """
        Extract contour from a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Contour points
        """
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Return the largest contour if found
        if contours:
            return max(contours, key=cv2.contourArea)
        else:
            return None


class SimpleSegmentor(BaseSegmentor):
    """Simple segmentor using OpenCV methods."""
    
    def __init__(self, config=None, system_info=None):
        """
        Initialize the simple segmentor.
        
        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)
    
    def segment(self, image, boxes):
        """
        Segment objects in an image based on bounding boxes.
        
        Args:
            image: Input image (numpy array)
            boxes: Bounding boxes (N, 4) where each box is (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing:
                - masks: Binary masks for each object
                - contours: Contour points for each object
        """
        height, width = image.shape[:2]
        masks = []
        contours = []
        
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box coordinates are valid
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                masks.append(np.zeros((height, width), dtype=bool))
                contours.append(None)
                continue
            
            # Extract object region
            roi = image[y1:y2, x1:x2]
            
            # Create mask
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            
            # Create contour
            contour = np.array([
                [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
            ], dtype=np.int32)
            contours.append(contour)
        
        return {
            'masks': masks,
            'contours': contours
        }
