import numpy as np
import logging
import cv2
from .base import BaseTracker
from postprocess.iou_tracker import associate_simple, box_iou
from .bytetrack_adapter import bytetrack_update, create_bytetrack

logger = logging.getLogger('object_detection.models.tracker')

class ObjectTracker(BaseTracker):
    """Class for tracking objects across video frames."""
    
    def __init__(self, config=None, system_info=None):
        """
        Initialize the object tracker.
        
        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        super().__init__(config, system_info)
        
        # Get tracker settings from config
        tracker_config = self.config.get('models', {}).get('tracker', {})
        
        self.tracker_type = tracker_config.get('type', 'bytetrack')
        self.track_thresh = tracker_config.get('track_thresh', 0.25)
        self.track_buffer = tracker_config.get('track_buffer', 30)
        self.match_thresh = tracker_config.get('match_thresh', 0.8)
        # Minimum IoU for the simple IoU tracker (match_thresh is ByteTrack's association threshold)
        self.simple_iou_thresh = tracker_config.get('simple_iou_thresh', 0.3)
        
        # Initialize tracker
        self.tracker = self._init_tracker()
        
        # Frame counter and tracks storage
        self.frame_count = 0
        self.tracks = {}  # Format: {track_id: {boxes: [], class: cls, first_frame: n, last_frame: m, ...}}
        self.next_id = 1  # For simple tracker
    
    def _init_tracker(self):
        """
        Initialize the appropriate tracker based on the configuration.
        
        Returns:
            Initialized tracker or None if using simple approach
        """
        if self.tracker_type == 'bytetrack':
            try:
                tracker = create_bytetrack(self.track_thresh, self.track_buffer, self.match_thresh, frame_rate=30)
                logger.info("ByteTrack initialized")
                return tracker
            except ImportError:
                logger.warning("ByteTrack not available (needs ultralytics), using simple tracker")
                return None
            except Exception as e:
                logger.error(f"Error initializing ByteTrack: {e}")
                return None

        elif self.tracker_type == 'sort':
            try:
                # Try to import SORT
                from sort.sort import Sort
                
                # Create tracker
                tracker = Sort(
                    max_age=self.track_buffer,
                    min_hits=3,
                    iou_threshold=self.match_thresh
                )
                
                logger.info("SORT initialized")
                return tracker
                
            except ImportError:
                logger.warning("SORT not available, using simple tracker")
                return None
            except Exception as e:
                logger.error(f"Error initializing SORT: {e}")
                return None
        
        elif self.tracker_type == 'simple':
            logger.info("Using simple IoU tracker")
            return None
        
        else:
            logger.warning(f"Unknown tracker type: {self.tracker_type}, using simple tracker")
            return None
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: Dictionary containing:
                - boxes: Bounding boxes (N, 4)
                - scores: Confidence scores (N,)
                - classes: Class IDs (N,)
        
        Returns:
            Dictionary containing:
                - tracks: Dictionary of track objects by ID
                - active_tracks: List of currently active track IDs
        """
        self.frame_count += 1
        
        # Handle empty detections
        if len(detections['boxes']) == 0:
            return {
                'tracks': self.tracks,
                'active_tracks': []
            }
        
        if self.tracker is None:
            # Use simple IoU-based tracking
            return self._update_simple(detections)
        elif self.tracker_type == 'bytetrack':
            # Use ByteTrack
            return self._update_bytetrack(detections)
        elif self.tracker_type == 'sort':
            # Use SORT
            return self._update_sort(detections)
        else:
            # Fallback to simple tracking
            return self._update_simple(detections)
    
    def _update_bytetrack(self, detections):
        """
        Update using ByteTrack.
        
        Args:
            detections: Detection results
            
        Returns:
            Tracking results
        """
        try:
            rows = bytetrack_update(self.tracker, detections['boxes'], detections['scores'], detections['classes'])
            active_tracks = []
            for row in rows:
                track_id = int(row[4])
                active_tracks.append(track_id)
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        'boxes': [],
                        'class': int(row[6]),
                        'first_frame': self.frame_count,
                        'last_frame': self.frame_count
                    }
                else:
                    self.tracks[track_id]['last_frame'] = self.frame_count
                self.tracks[track_id]['boxes'].append(np.asarray(row[:4], dtype=float))

            return {
                'tracks': self.tracks,
                'active_tracks': active_tracks
            }

        except Exception as e:
            logger.error(f"Error in ByteTrack update: {e}")
            # Fallback to simple tracking
            return self._update_simple(detections)

    def _update_sort(self, detections):
        """
        Update using SORT.
        
        Args:
            detections: Detection results
            
        Returns:
            Tracking results
        """
        try:
            # Convert detections to format expected by SORT: [x1, y1, x2, y2, score]
            boxes = detections['boxes']
            scores = detections['scores']
            classes = detections['classes']
            
            # Combine into single array (SORT doesn't use class info)
            dets = np.concatenate([boxes, scores[:, None]], axis=1)
            
            # Update tracker
            tracks = self.tracker.update(dets)
            
            # Process tracker results
            active_tracks = []
            
            for t in tracks:
                track_id = int(t[4])
                active_tracks.append(track_id)
                
                # Create or update track
                if track_id not in self.tracks:
                    # Get class from the detection that matches this track
                    box = t[:4]
                    
                    # Find the most overlapping detection
                    max_iou = 0
                    cls = 0
                    
                    for i, det_box in enumerate(boxes):
                        iou = self._calculate_iou(box, det_box)
                        if iou > max_iou:
                            max_iou = iou
                            cls = classes[i]
                    
                    self.tracks[track_id] = {
                        'boxes': [],
                        'class': int(cls),
                        'first_frame': self.frame_count,
                        'last_frame': self.frame_count
                    }
                else:
                    self.tracks[track_id]['last_frame'] = self.frame_count
                
                # Add current position
                box = t[:4]  # [x1, y1, x2, y2]
                self.tracks[track_id]['boxes'].append(box)
            
            return {
                'tracks': self.tracks,
                'active_tracks': active_tracks
            }
            
        except Exception as e:
            logger.error(f"Error in SORT update: {e}")
            # Fallback to simple tracking
            return self._update_simple(detections)
    
    def _update_simple(self, detections):
        """
        Simple IoU-based tracking (postprocess/iou_tracker.associate_simple).

        Args:
            detections: Detection results

        Returns:
            Tracking results
        """
        active_tracks, self.next_id = associate_simple(
            self.tracks, detections['boxes'], detections['scores'], detections['classes'],
            self.frame_count, self.next_id, self.track_buffer, self.simple_iou_thresh)
        return {
            'tracks': self.tracks,
            'active_tracks': active_tracks
        }

    def _calculate_iou(self, box1, box2):
        """IoU of two [x1, y1, x2, y2] boxes (postprocess/iou_tracker.box_iou)."""
        return box_iou(box1, box2)

    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        self.tracks = {}
        self.next_id = 1
        
        # Re-initialize tracker if needed
        if self.tracker_type == 'bytetrack':
            try:
                self.tracker = create_bytetrack(self.track_thresh, self.track_buffer, self.match_thresh, frame_rate=30)
            except Exception:
                self.tracker = None

        elif self.tracker_type == 'sort':
            try:
                from sort.sort import Sort
                self.tracker = Sort(
                    max_age=self.track_buffer,
                    min_hits=3,
                    iou_threshold=self.match_thresh
                )
            except:
                self.tracker = None
