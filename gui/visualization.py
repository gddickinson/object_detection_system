from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, 
                           QSizePolicy, QGraphicsView, QGraphicsScene, 
                           QGraphicsItem, QMenu, QAction)
from PyQt5.QtCore import Qt, QRectF, QPointF, QPoint
from PyQt5.QtGui import (QImage, QPixmap, QColor, QPainter, QPen, QBrush, 
                       QFont, QCursor, QTransform, QWheelEvent, QPainterPath)
import cv2
import numpy as np
import logging
import json
import random
import os

logger = logging.getLogger('object_detection.gui.visualization')

class ZoomableGraphicsView(QGraphicsView):
    """
    Zoomable graphics view for image visualization.
    Allows zooming, panning, and context menu.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(400, 300)
        
        # Set up scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set up zooming variables
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_zoom = 1.0
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming.
        
        Args:
            event: Wheel event
        """
        # Zoom factor
        zoom_in = event.angleDelta().y() > 0
        
        if zoom_in:
            # Zoom in
            factor = self.zoom_factor
        else:
            # Zoom out
            factor = 1.0 / self.zoom_factor
        
        # Calculate new zoom level
        new_zoom = self.current_zoom * factor
        
        # Clamp zoom level
        if new_zoom < self.min_zoom:
            factor = self.min_zoom / self.current_zoom
            new_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            factor = self.max_zoom / self.current_zoom
            new_zoom = self.max_zoom
        
        # Update current zoom
        self.current_zoom = new_zoom
        
        # Apply zoom
        self.scale(factor, factor)
        
        # Accept the event
        event.accept()
    
    def reset_zoom(self):
        """Reset zoom to original size."""
        # Calculate factor to reset zoom
        factor = 1.0 / self.current_zoom
        
        # Reset transform
        self.setTransform(QTransform())
        
        # Reset current zoom
        self.current_zoom = 1.0
    
    def fit_to_view(self):
        """Fit the entire scene in the view."""
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # Update current zoom
        transform = self.transform()
        self.current_zoom = (transform.m11() + transform.m22()) / 2.0
    
    def show_context_menu(self, position):
        """
        Show context menu at the given position.
        
        Args:
            position: Menu position
        """
        menu = QMenu()
        
        # Reset zoom action
        reset_action = QAction("Reset Zoom", self)
        reset_action.triggered.connect(self.reset_zoom)
        menu.addAction(reset_action)
        
        # Fit to view action
        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(self.fit_to_view)
        menu.addAction(fit_action)
        
        # Show menu
        menu.exec_(self.mapToGlobal(position))


class VisualizationWidget(QWidget):
    """Widget for visualizing detection, segmentation, and tracking results."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results = None
        self.track_colors = {}
        self.scale_factor = 1.0
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Graphics view for image display
        self.graphics_view = ZoomableGraphicsView()
        layout.addWidget(self.graphics_view)
        
        # Set minimum size
        self.setMinimumSize(600, 400)
    
    def update_visualization(self, results):
        """
        Update the visualization with new results.
        
        Args:
            results: Dictionary containing detection, segmentation, and tracking results
        """
        self.results = results
        self.draw_results()
    
    def draw_results(self):
        """Draw detection, segmentation, and tracking results on the image."""
        if self.results is None:
            return
        
        # Clear the scene
        self.graphics_view.scene.clear()
        
        # Get the image
        image = self.results['image'].copy()
        
        # Create a copy for drawing on
        vis_image = image.copy()
        
        # Draw segmentations if enabled
        if self.results.get('show_segmentation', True) and 'segmentations' in self.results:
            vis_image = self.draw_segmentations(vis_image, self.results['segmentations'])
        
        # Draw detections
        if 'detections' in self.results:
            vis_image = self.draw_detections(vis_image, self.results['detections'])
        
        # Draw tracking info
        if self.results.get('show_tracking', True) and 'tracking' in self.results:
            vis_image = self.draw_tracking(vis_image, self.results['tracking'], self.results['detections'])
        
        # Draw frame info for videos
        if 'frame_index' in self.results and 'total_frames' in self.results:
            frame_text = f"Frame: {self.results['frame_index']+1} / {self.results['total_frames']}"
            cv2.putText(vis_image, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert to QImage and display
        height, width, channel = vis_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(vis_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Create pixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Add to scene
        self.graphics_view.scene.addPixmap(pixmap)
        self.graphics_view.scene.setSceneRect(0, 0, width, height)
        
        # Fit to view on first image
        if self.results.get('frame_index', 0) == 0:
            self.graphics_view.fit_to_view()
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels for detected objects.
        
        Args:
            image: Image to draw on
            detections: Detection results
            
        Returns:
            Image with detections drawn
        """
        show_labels = self.results.get('show_labels', True)
        
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class info
            class_name = detections['class_names'][i]
            score = detections['scores'][i]
            
            # Get color based on class
            color = self.get_color_for_class(class_name)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label if enabled
            if show_labels:
                # Prepare label text
                label = f"{class_name}: {score:.2f}"
                if 'llm_descriptions' in detections:
                    label = f"{detections['llm_descriptions'][i]}: {score:.2f}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = map(int, detections['centers'][i])
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Show coordinates
            if show_labels:
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(image, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return image
    
    def draw_segmentations(self, image, segmentations):
        """
        Draw segmentation masks and contours.
        
        Args:
            image: Image to draw on
            segmentations: Segmentation results
            
        Returns:
            Image with segmentations drawn
        """
        # Create a copy of the image for blending
        overlay = image.copy()
        
        # Draw masks
        for i, mask in enumerate(segmentations['masks']):
            # Skip empty masks
            if mask.size == 0:
                continue
            
            # Convert mask to correct shape if needed
            if mask.shape != (image.shape[0], image.shape[1]):
                # This might happen if masks are returned in a different size
                # Resize to match the image
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                mask = mask > 0
            
            # Color for this instance (use random color)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Apply mask overlay
            mask_binary = mask.astype(np.uint8)
            colored_mask = np.zeros_like(image)
            colored_mask[mask_binary == 1] = color
            
            # Blend mask with image
            alpha = 0.3
            cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Draw contour
            if segmentations['contours'][i] is not None:
                cv2.drawContours(overlay, [segmentations['contours'][i]], 0, color, 2)
        
        return overlay
    
    def draw_tracking(self, image, tracking, detections):
        """
        Draw tracking information.
        
        Args:
            image: Image to draw on
            tracking: Tracking results
            detections: Detection results
            
        Returns:
            Image with tracking drawn
        """
        show_labels = self.results.get('show_labels', True)
        
        # Assign colors to tracks
        for track_id in tracking['active_tracks']:
            if track_id not in self.track_colors:
                self.track_colors[track_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
        
        # Draw active tracks
        for track_id in tracking['active_tracks']:
            if track_id in tracking['tracks']:
                track = tracking['tracks'][track_id]
                
                # Get the latest box
                if track['boxes']:
                    box = track['boxes'][-1]
                    if len(box) >= 4:
                        x1, y1, x2, y2 = map(int, box[:4])
                        
                        # Draw bounding box
                        color = self.track_colors[track_id]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw ID
                        if show_labels:
                            id_text = f"ID: {track_id}"
                            cv2.putText(image, id_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw trajectory
                        if len(track['boxes']) > 1:
                            pts = []
                            for j in range(min(20, len(track['boxes']))):  # Limit to last 20 frames
                                idx = len(track['boxes']) - j - 1
                                prev_box = track['boxes'][idx]
                                if len(prev_box) >= 4:
                                    prev_x1, prev_y1 = prev_box[0], prev_box[1]
                                    prev_x2, prev_y2 = prev_box[2], prev_box[3]
                                    
                                    # Get center point of box
                                    center_x = int((prev_x1 + prev_x2) / 2)
                                    center_y = int((prev_y1 + prev_y2) / 2)
                                    
                                    pts.append((center_x, center_y))
                            
                            # Draw trajectory line
                            for j in range(1, len(pts)):
                                cv2.line(image, pts[j-1], pts[j], color, 2)
        
        return image
    
    def get_color_for_class(self, class_name):
        """
        Get a consistent color for a class.
        
        Args:
            class_name: Class name
            
        Returns:
            BGR color tuple
        """
        # Hash the class name to get a consistent color
        hash_val = hash(class_name) % 255
        
        # Generate a color using HSV (more visually distinct colors)
        h = hash_val / 255.0
        s = 0.8
        v = 0.8
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to 0-255 range and BGR order
        return (int(b * 255), int(g * 255), int(r * 255))
    
    def reset_view(self):
        """Reset the view to its original state."""
        if self.graphics_view:
            self.graphics_view.reset_zoom()
            
            # Re-fit if there's content
            if not self.graphics_view.scene.items():
                self.graphics_view.fit_to_view()
    
    def save_current_view(self, file_path):
        """
        Save the current view to an image file.
        
        Args:
            file_path: Path to save the image to
        """
        if self.results is None:
            return
        
        # Get the image
        image = self.results['image'].copy()
        
        # Create a copy for drawing on
        vis_image = image.copy()
        
        # Draw segmentations
        if self.results.get('show_segmentation', True) and 'segmentations' in self.results:
            vis_image = self.draw_segmentations(vis_image, self.results['segmentations'])
        
        # Draw detections
        if 'detections' in self.results:
            vis_image = self.draw_detections(vis_image, self.results['detections'])
        
        # Draw tracking info
        if self.results.get('show_tracking', True) and 'tracking' in self.results:
            vis_image = self.draw_tracking(vis_image, self.results['tracking'], self.results['detections'])
        
        # Draw frame info for videos
        if 'frame_index' in self.results and 'total_frames' in self.results:
            frame_text = f"Frame: {self.results['frame_index']+1} / {self.results['total_frames']}"
            cv2.putText(vis_image, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save image
        cv2.imwrite(file_path, vis_image)
    
    def save_results_json(self, file_path):
        """
        Save the current results to a JSON file.
        
        Args:
            file_path: Path to save the JSON to
        """
        if self.results is None:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        
        if 'detections' in self.results:
            json_results['detections'] = {
                'boxes': self.results['detections']['boxes'].tolist() if len(self.results['detections']['boxes']) > 0 else [],
                'scores': self.results['detections']['scores'].tolist() if len(self.results['detections']['scores']) > 0 else [],
                'classes': self.results['detections']['classes'].tolist() if len(self.results['detections']['classes']) > 0 else [],
                'class_names': self.results['detections']['class_names'],
                'centers': self.results['detections']['centers'].tolist() if len(self.results['detections']['centers']) > 0 else []
            }
            
            if 'llm_descriptions' in self.results['detections']:
                json_results['detections']['llm_descriptions'] = self.results['detections']['llm_descriptions']
        
        if 'tracking' in self.results:
            # Convert track data
            tracks_json = {}
            
            for track_id, track in self.results['tracking']['tracks'].items():
                tracks_json[str(track_id)] = {
                    'boxes': [box.tolist() if hasattr(box, 'tolist') else box for box in track['boxes']],
                    'class': track.get('class', 0),
                    'first_frame': track.get('first_frame', 0),
                    'last_frame': track.get('last_frame', 0)
                }
            
            json_results['tracking'] = {
                'tracks': tracks_json,
                'active_tracks': self.results['tracking']['active_tracks']
            }
        
        if 'frame_index' in self.results:
            json_results['frame_index'] = self.results['frame_index']
        
        if 'total_frames' in self.results:
            json_results['total_frames'] = self.results['total_frames']
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2)
