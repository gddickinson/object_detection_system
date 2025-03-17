from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QComboBox,
                            QSlider, QProgressBar, QStatusBar, QAction, QMenu,
                            QToolBar, QDockWidget, QTabWidget, QMessageBox,
                            QSplitter, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QSize
from PyQt5.QtGui import QIcon, QPixmap, QImage, QKeySequence
import cv2
import numpy as np
import yaml
import os
import logging
import time

from models.detector import create_detector
from models.segmentor import ObjectSegmentor
from models.tracker import ObjectTracker
from models.llm_integration import LLMAnalyzer
from gui.visualization import VisualizationWidget
from gui.settings_dialog import SettingsDialog
from auth.user_manager import UserManagementDialog
from plugins.plugin_manager import PluginManager
from utils.system_info import get_optimal_batch_size

logger = logging.getLogger('object_detection.gui')

class ProcessingThread(QThread):
    """Thread for processing images and videos without blocking the GUI."""
    update_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, input_path, models, is_video=False, config=None, system_info=None):
        super().__init__()
        self.input_path = input_path
        self.detector = models['detector']
        self.segmentor = models['segmentor']
        self.tracker = models['tracker']
        self.llm_analyzer = models['llm_analyzer']
        self.is_video = is_video
        self.running = True
        self.config = config or {}
        self.system_info = system_info or {}

        # Get processing settings
        processing_config = self.config.get('processing', {})
        self.batch_size = processing_config.get('batch_size', 1)

        # Adjust batch size based on system capabilities
        if system_info:
            optimal_batch_size = get_optimal_batch_size(system_info)
            self.batch_size = min(self.batch_size, optimal_batch_size)

        self.save_results = processing_config.get('save_results', True)
        self.output_dir = processing_config.get('output_dir', 'output')

        # Create output directory if needed
        if self.save_results and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Run image or video processing."""
        try:
            if self.is_video:
                self._process_video()
            else:
                self._process_image()

            self.finished_signal.emit()
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            self.error_signal.emit(f"Processing error: {str(e)}")
            self.finished_signal.emit()

    def _process_image(self):
        """Process a single image."""
        try:
            image = cv2.imread(self.input_path)
            if image is None:
                self.error_signal.emit(f"Failed to read image: {self.input_path}")
                return

            # Object detection
            detections = self.detector.detect(image)

            # Object segmentation
            if len(detections['boxes']) > 0:
                segmentations = self.segmentor.segment(image, detections['boxes'])
            else:
                segmentations = {'masks': [], 'contours': []}

            # Combine results
            results = {
                'image': image,
                'detections': detections,
                'segmentations': segmentations,
                'frame_index': 0,
                'is_video': False
            }

            # Optionally add LLM analysis
            if self.llm_analyzer:
                results['detections'] = self.llm_analyzer.analyze_objects(image, detections)

            self.update_signal.emit(results)

            # Save results if enabled
            if self.save_results:
                output_path = os.path.join(self.output_dir, os.path.basename(self.input_path))
                self._save_results(results, output_path)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.error_signal.emit(f"Error processing image: {str(e)}")

    def _process_video(self):
        """Process a video file."""
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.error_signal.emit(f"Failed to open video: {self.input_path}")
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup video writer if saving results
            if self.save_results:
                output_path = os.path.join(self.output_dir, os.path.basename(self.input_path))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Process frames
            current_frame = 0

            while self.running and current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Object detection
                detections = self.detector.detect(frame)

                # Object tracking
                tracking = self.tracker.update(detections)

                # Object segmentation (only for active tracks to save processing)
                if len(detections['boxes']) > 0:
                    segmentations = self.segmentor.segment(frame, detections['boxes'])
                else:
                    segmentations = {'masks': [], 'contours': []}

                # Combine results
                results = {
                    'image': frame,
                    'detections': detections,
                    'segmentations': segmentations,
                    'tracking': tracking,
                    'frame_index': current_frame,
                    'total_frames': total_frames,
                    'is_video': True
                }

                # Optionally add LLM analysis (for keyframes only to save time)
                if self.llm_analyzer and current_frame % 30 == 0:  # Every 30 frames
                    results['detections'] = self.llm_analyzer.analyze_objects(frame, detections)

                self.update_signal.emit(results)

                # Save frame if enabled
                if self.save_results:
                    annotated_frame = self._annotate_frame(results)
                    out.write(annotated_frame)

                # Update progress
                progress = int((current_frame / total_frames) * 100)
                self.progress_signal.emit(progress)

                current_frame += 1

            # Clean up
            cap.release()
            if self.save_results:
                out.release()

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            self.error_signal.emit(f"Error processing video: {str(e)}")

    def _annotate_frame(self, results):
        """
        Annotate a frame with detection and tracking results.

        Args:
            results: Results dictionary

        Returns:
            Annotated frame
        """
        # Create a copy of the frame
        frame = results['image'].copy()

        # Draw detections
        for i, box in enumerate(results['detections']['boxes']):
            x1, y1, x2, y2 = map(int, box)

            # Get class name and score
            class_name = results['detections']['class_names'][i]
            score = results['detections']['scores'][i]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {score:.2f}"
            if 'llm_descriptions' in results['detections']:
                label = f"{results['detections']['llm_descriptions'][i]}: {score:.2f}"

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw tracking info if available
        if 'tracking' in results:
            for track_id in results['tracking']['active_tracks']:
                if track_id in results['tracking']['tracks']:
                    track = results['tracking']['tracks'][track_id]

                    # Get the latest box
                    if track['boxes']:
                        box = track['boxes'][-1]
                        if len(box) >= 4:
                            x1, y1, x2, y2 = map(int, box[:4])

                            # Draw ID
                            cv2.putText(frame, f"ID: {track_id}",
                                      (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

    def _save_results(self, results, output_path):
        """
        Save processing results.

        Args:
            results: Results dictionary
            output_path: Path to save the results
        """
        # Save annotated image
        annotated_image = self._annotate_frame(results)
        cv2.imwrite(output_path, annotated_image)

        # Save JSON results
        json_path = os.path.splitext(output_path)[0] + '.json'

        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'detections': {
                'boxes': results['detections']['boxes'].tolist() if len(results['detections']['boxes']) > 0 else [],
                'scores': results['detections']['scores'].tolist() if len(results['detections']['scores']) > 0 else [],
                'classes': results['detections']['classes'].tolist() if len(results['detections']['classes']) > 0 else [],
                'class_names': results['detections']['class_names'],
                'centers': results['detections']['centers'].tolist() if len(results['detections']['centers']) > 0 else []
            }
        }

        # Add LLM descriptions if available
        if 'llm_descriptions' in results['detections']:
            json_results['detections']['llm_descriptions'] = results['detections']['llm_descriptions']

        import json
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

    def stop(self):
        """Stop processing."""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, args=None, config=None, system_info=None, user=None):
        super().__init__()
        self.args = args
        self.config = config or {}
        self.system_info = system_info or {}
        self.user = user

        self.models = {
            'detector': None,
            'segmentor': None,
            'tracker': None,
            'llm_analyzer': None
        }

        self.processing_thread = None
        self.current_input = None
        self.is_video = False
        self.settings = QSettings("ObjectDetectionApp", "MainWindow")

        # Load plugins
        self.plugin_manager = PluginManager(config)

        self.init_ui()
        self.init_models()

        # Load initial input if provided
        if args and args.input:
            self.current_input = args.input
            self.is_video = self.current_input.lower().endswith(('.mp4', '.avi', '.mov'))
            self.status_bar.showMessage(f"Loaded input: {self.current_input}")

            # Automatically process if no-gui mode
            if args.no_gui:
                self.process_input()



    def init_ui(self):
        """Initialize the user interface."""
        # Get GUI settings
        gui_config = self.config.get('gui', {})
        width = gui_config.get('width', 1200)
        height = gui_config.get('height', 800)

        self.setWindowTitle("Object Detection and Tracking System")
        self.setGeometry(100, 100, width, height)

        # Main widget - DON'T INITIALIZE MENU HERE
        # self.init_menu()
        # self.init_toolbar()

        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Splitter for control panel and visualization
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # Control panel
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        # Input controls
        input_layout = QHBoxLayout()

        self.input_btn = QPushButton("Select Input")
        self.input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_btn)

        self.input_label = QLabel("No input selected")
        input_layout.addWidget(self.input_label, 1)

        control_layout.addLayout(input_layout)

        # Model and settings controls
        settings_layout = QHBoxLayout()

        # Model selection
        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        settings_layout.addWidget(self.model_combo)

        # Confidence threshold
        settings_layout.addWidget(QLabel("Confidence:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        settings_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.50")
        settings_layout.addWidget(self.threshold_label)

        # LLM integration
        settings_layout.addWidget(QLabel("LLM:"))
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(["None", "Ollama", "Cloud API"])
        self.llm_combo.setCurrentText("Ollama")  # Default to Ollama
        settings_layout.addWidget(self.llm_combo)

        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_input)
        settings_layout.addWidget(self.process_btn)

        control_layout.addLayout(settings_layout)

        # Options
        options_layout = QHBoxLayout()

        # Save results
        self.save_results_check = QCheckBox("Save Results")
        self.save_results_check.setChecked(True)
        options_layout.addWidget(self.save_results_check)

        # Show segmentation
        self.show_segmentation_check = QCheckBox("Show Segmentation")
        self.show_segmentation_check.setChecked(True)
        options_layout.addWidget(self.show_segmentation_check)

        # Show tracking
        self.show_tracking_check = QCheckBox("Show Tracking")
        self.show_tracking_check.setChecked(True)
        options_layout.addWidget(self.show_tracking_check)

        # Show labels
        self.show_labels_check = QCheckBox("Show Labels")
        self.show_labels_check.setChecked(True)
        options_layout.addWidget(self.show_labels_check)

        control_layout.addLayout(options_layout)

        # Add control widget to splitter
        splitter.addWidget(control_widget)

        # Visualization widget
        self.vis_widget = VisualizationWidget()
        splitter.addWidget(self.vis_widget)

        # Set splitter proportions
        splitter.setSizes([200, 600])

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Show user info if logged in
        if self.user:
            self.status_bar.showMessage(f"Logged in as: {self.user.username} ({self.user.role})")
        else:
            self.status_bar.showMessage("Ready")

        # NOW initialize menu and toolbar AFTER vis_widget is created
        self.init_menu()
        self.init_toolbar()

        # Restore window state if available
        self.restore_window_state()

    def init_menu(self):
        """Initialize application menu."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Input", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.select_input)
        file_menu.addAction(open_action)

        save_action = QAction("Save Results", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        edit_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("View")

        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.vis_widget.reset_view)
        view_menu.addAction(reset_view_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        if self.user and self.user.is_admin:
            user_mgmt_action = QAction("User Management", self)
            user_mgmt_action.triggered.connect(self.show_user_management)
            tools_menu.addAction(user_mgmt_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def init_toolbar(self):
        """Initialize toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open input action
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.select_input)
        toolbar.addAction(open_action)

        # Process action
        process_action = QAction("Process", self)
        process_action.triggered.connect(self.process_input)
        toolbar.addAction(process_action)

        # Stop action
        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_processing)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        # Save results action
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)

    def restore_window_state(self):
        """Restore window state from settings."""
        # Check if remember settings is enabled
        if self.config.get('gui', {}).get('remember_settings', True):
            # Restore window geometry
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)

            # Restore window state
            state = self.settings.value("windowState")
            if state:
                self.restoreState(state)

            # Restore other settings
            confidence = self.settings.value("confidence", 50, type=int)
            self.threshold_slider.setValue(confidence)

            llm_type = self.settings.value("llm_type", "Ollama")
            index = self.llm_combo.findText(llm_type)
            if index >= 0:
                self.llm_combo.setCurrentIndex(index)

            save_results = self.settings.value("save_results", True, type=bool)
            self.save_results_check.setChecked(save_results)

            show_segmentation = self.settings.value("show_segmentation", True, type=bool)
            self.show_segmentation_check.setChecked(show_segmentation)

            show_tracking = self.settings.value("show_tracking", True, type=bool)
            self.show_tracking_check.setChecked(show_tracking)

            show_labels = self.settings.value("show_labels", True, type=bool)
            self.show_labels_check.setChecked(show_labels)

    def save_window_state(self):
        """Save window state to settings."""
        # Check if remember settings is enabled
        if self.config.get('gui', {}).get('remember_settings', True):
            # Save window geometry
            self.settings.setValue("geometry", self.saveGeometry())

            # Save window state
            self.settings.setValue("windowState", self.saveState())

            # Save other settings
            self.settings.setValue("confidence", self.threshold_slider.value())
            self.settings.setValue("llm_type", self.llm_combo.currentText())
            self.settings.setValue("save_results", self.save_results_check.isChecked())
            self.settings.setValue("show_segmentation", self.show_segmentation_check.isChecked())
            self.settings.setValue("show_tracking", self.show_tracking_check.isChecked())
            self.settings.setValue("show_labels", self.show_labels_check.isChecked())

    def closeEvent(self, event):
        """Handle window close event."""
        # Save window state
        self.save_window_state()

        # Stop processing if running
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()

        # Accept the event
        event.accept()

    def update_threshold_label(self, value):
        """Update the threshold label when the slider changes."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")

        # Update detector confidence if initialized
        if self.models['detector']:
            self.models['detector'].set_confidence(threshold)

    def select_input(self):
        """Open file dialog to select input image or video."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images & Videos (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.current_input = selected_files[0]
                self.is_video = self.current_input.lower().endswith(('.mp4', '.avi', '.mov'))
                self.status_bar.showMessage(f"Selected: {self.current_input}")
                self.input_label.setText(os.path.basename(self.current_input))

    def init_models(self):
        """Initialize AI models."""
        try:
            # Initialize detector
            self.models['detector'] = create_detector(self.config, self.system_info)

            # Set confidence from slider
            if self.models['detector']:
                threshold = self.threshold_slider.value() / 100.0
                self.models['detector'].set_confidence(threshold)

            # Initialize segmentor
            self.models['segmentor'] = ObjectSegmentor(self.config, self.system_info)

            # Initialize tracker
            self.models['tracker'] = ObjectTracker(self.config, self.system_info)

            # Initialize LLM analyzer based on selected type
            llm_type = self.llm_combo.currentText()

            if llm_type == "None":
                self.models['llm_analyzer'] = None
            else:
                # Update config with selected LLM type
                if llm_type == "Ollama":
                    self.config.setdefault('llm', {})['type'] = 'ollama'
                elif llm_type == "Cloud API":
                    self.config.setdefault('llm', {})['type'] = 'cloud'

                self.models['llm_analyzer'] = LLMAnalyzer(self.config, self.system_info)

            self.status_bar.showMessage("Models initialized")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            QMessageBox.critical(self, "Error", f"Error initializing models: {str(e)}")

    def process_input(self):
        """Process the selected input file."""
        if not self.current_input:
            self.status_bar.showMessage("No input selected")
            return

        try:
            # Update confidence threshold
            threshold = self.threshold_slider.value() / 100.0
            if self.models['detector']:
                self.models['detector'].set_confidence(threshold)

            # Update LLM analyzer based on selected type
            llm_type = self.llm_combo.currentText()

            if llm_type == "None":
                self.models['llm_analyzer'] = None
            else:
                # Update config with selected LLM type
                if llm_type == "Ollama":
                    self.config.setdefault('llm', {})['type'] = 'ollama'
                elif llm_type == "Cloud API":
                    self.config.setdefault('llm', {})['type'] = 'cloud'

                # Check if analyzer needs to be created or updated
                if not self.models['llm_analyzer'] or self.models['llm_analyzer'].llm_type != self.config['llm']['type']:
                    self.models['llm_analyzer'] = LLMAnalyzer(self.config, self.system_info)

            # Update processing settings
            self.config.setdefault('processing', {})['save_results'] = self.save_results_check.isChecked()

            # Reset tracker for new input
            if self.models['tracker']:
                self.models['tracker'].reset()

            # Start processing thread
            self.processing_thread = ProcessingThread(
                self.current_input,
                self.models,
                self.is_video,
                self.config,
                self.system_info
            )

            # Connect signals
            self.processing_thread.update_signal.connect(self.update_visualization)
            self.processing_thread.progress_signal.connect(self.update_progress)
            self.processing_thread.finished_signal.connect(self.processing_finished)
            self.processing_thread.error_signal.connect(self.show_error)

            # Start processing
            self.progress_bar.setVisible(True)
            self.process_btn.setEnabled(False)
            self.processing_thread.start()

            self.status_bar.showMessage("Processing...")

        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            QMessageBox.critical(self, "Error", f"Error starting processing: {str(e)}")

    def update_visualization(self, results):
        """Update visualization with new results."""
        # Update visualization options
        results['show_segmentation'] = self.show_segmentation_check.isChecked()
        results['show_tracking'] = self.show_tracking_check.isChecked()
        results['show_labels'] = self.show_labels_check.isChecked()

        self.vis_widget.update_visualization(results)

    def update_progress(self, progress):
        """Update progress bar."""
        self.progress_bar.setValue(progress)

    def processing_finished(self):
        """Handle processing thread completion."""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Processing completed")

    def show_error(self, error_message):
        """Show error message."""
        QMessageBox.critical(self, "Error", error_message)

    def stop_processing(self):
        """Stop the processing thread."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_bar.showMessage("Processing stopped")

    def save_results(self):
        """Save the current results."""
        if not self.vis_widget.results:
            QMessageBox.warning(self, "No Results", "No results to save.")
            return

        # Get output directory
        output_dir = self.config.get('processing', {}).get('output_dir', 'output')

        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Get base name of input file
        if self.current_input:
            base_name = os.path.basename(self.current_input)
            output_path = os.path.join(output_dir, base_name)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"result_{timestamp}.jpg")

        try:
            # Save annotated image
            self.vis_widget.save_current_view(output_path)

            # Save JSON results
            json_path = os.path.splitext(output_path)[0] + '.json'
            self.vis_widget.save_results_json(json_path)

            self.status_bar.showMessage(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")

    def show_settings_dialog(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self, self.config)
        if dialog.exec_():
            # Update config
            self.config = dialog.get_config()

            # Apply settings
            self.apply_settings()

    def apply_settings(self):
        """Apply settings from config."""
        # Update GUI settings
        gui_config = self.config.get('gui', {})
        font_size = gui_config.get('font_size', 10)
        self.setStyleSheet(f"font-size: {font_size}pt;")

        # Update LLM settings
        llm_config = self.config.get('llm', {})
        llm_type = llm_config.get('type', 'none')

        if llm_type == 'none':
            self.llm_combo.setCurrentText("None")
        elif llm_type == 'ollama':
            self.llm_combo.setCurrentText("Ollama")
        elif llm_type == 'cloud':
            self.llm_combo.setCurrentText("Cloud API")

        # Reinitialize models if needed
        self.init_models()

    def show_user_management(self):
        """Show user management dialog."""
        if not self.user or not self.user.is_admin:
            QMessageBox.warning(self, "Permission Denied", "Admin rights required to access user management.")
            return

        dialog = UserManagementDialog(self, self.config)
        dialog.exec_()

    def show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(self, "About",
                        "Object Detection and Tracking System\n\n"
                        "A comprehensive tool for detecting and tracking objects in images and videos.\n\n"
                        "Version: 1.0\n"
                        "© 2025")
