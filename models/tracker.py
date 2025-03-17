import numpy as np
import logging
import cv2
from .base import BaseTracker

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
                # Try to import ByteTrack
                from ultralytics.trackers.byte_track.byte_tracker import BYTETracker
                
                # Create tracker
                tracker = BYTETracker(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=30
                )
                
                logger.info("ByteTrack initialized")
                return tracker
                
            except ImportError:
                logger.warning("ByteTrack not available, using simple tracker")
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
            # Convert detections to format expected by ByteTrack: [x1, y1, x2, y2, score, class]
            boxes = detections['boxes']
            scores = detections['scores']
            classes = detections['classes']
            
            # Combine into single array
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=1)
            
            # Update tracker
            online_targets = self.tracker.update(dets, (1, 1), (1, 1))  # Dummy image size
            
            # Process tracker results
            active_tracks = []
            
            for t in online_targets:
                track_id = t.track_id
                active_tracks.append(track_id)
                
                # Create or update track
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        'boxes': [],
                        'class': int(t.cls),
                        'first_frame': self.frame_count,
                        'last_frame': self.frame_count
                    }
                else:
                    self.tracks[track_id]['last_frame'] = self.frame_count
                
                # Add current position
                box = t.tlbr  # top-left, bottom-right format
                self.tracks[track_id]['boxes'].append(box)
            
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
        Simple IoU-based tracking.
        
        Args:
            detections: Detection results
            
        Returns:
            Tracking results
        """
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']
        
        # No detections, just return current tracks
        if len(boxes) == 0:
            return {
                'tracks': self.tracks,
                'active_tracks': []
            }
        
        # List of active tracks for this frame
        active_tracks = []
        
        # Store matched tracks to avoid double-matching
        matched_tracks = set()
        
        # Store matched detections to avoid double-matching
        matched_detections = set()
        
        # Find matches between existing tracks and new detections
        for track_id, track in self.tracks.items():
            # Skip if track is too old (inactive)
            if (self.frame_count - track['last_frame']) > self.track_buffer:
                continue
            
            # Get the last box of this track
            if not track['boxes']:
                continue
                
            last_box = track['boxes'][-1]
            
            # Calculate IoU with each detection
            best_iou = self.match_thresh
            best_idx = -1
            
            for i, box in enumerate(boxes):
                # Skip if this detection was already matched
                if i in matched_detections:
                    continue
                
                # Skip if class doesn't match
                if track['class'] != classes[i]:
                    continue
                
                iou = self._calculate_iou(last_box, box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # If a match was found, update the track
            if best_idx >= 0:
                # Mark as matched
                matched_tracks.add(track_id)
                matched_detections.add(best_idx)
                
                # Update track
                track['boxes'].append(boxes[best_idx])
                track['last_frame'] = self.frame_count
                
                # Add to active tracks
                active_tracks.append(track_id)
        
        # Create new tracks for unmatched detections
        for i, box in enumerate(boxes):
            if i not in matched_detections:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'boxes': [box],
                    'class': int(classes[i]),
                    'score': float(scores[i]),
                    'first_frame': self.frame_count,
                    'last_frame': self.frame_count
                }
                
                # Add to active tracks
                active_tracks.append(track_id)
        
        return {
            'tracks': self.tracks,
            'active_tracks': active_tracks
        }
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        if union_area > 0:
            return intersection_area / union_area
        else:
            return 0.0
    
    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        self.tracks = {}
        self.next_id = 1
        
        # Re-initialize tracker if needed
        if self.tracker_type == 'bytetrack':
            try:
                from ultralytics.trackers.byte_track.byte_tracker import BYTETracker
                self.tracker = BYTETracker(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=30
                )
            except:
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
