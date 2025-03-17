from PyQt5.QtWidgets import (QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, 
                           QPushButton, QFormLayout, QGroupBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt
import yaml
import os
import logging

logger = logging.getLogger('object_detection.gui.settings')

class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 400)
        
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Tab widget for settings categories
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_general_tab()
        self.create_models_tab()
        self.create_llm_tab()
        self.create_processing_tab()
        self.create_system_tab()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        export_button = QPushButton("Export")
        export_button.clicked.connect(self.export_settings)
        button_layout.addWidget(export_button)
        
        import_button = QPushButton("Import")
        import_button.clicked.connect(self.import_settings)
        button_layout.addWidget(import_button)
        
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
    
    def create_general_tab(self):
        """Create the general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # GUI settings
        gui_group = QGroupBox("GUI Settings")
        gui_layout = QFormLayout()
        
        # Window size
        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 3840)
        self.width_spin.setSingleStep(100)
        gui_layout.addRow("Window Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 2160)
        self.height_spin.setSingleStep(100)
        gui_layout.addRow("Window Height:", self.height_spin)
        
        # Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["system", "light", "dark"])
        gui_layout.addRow("Theme:", self.theme_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        gui_layout.addRow("Font Size:", self.font_size_spin)
        
        # Remember settings
        self.remember_check = QCheckBox()
        gui_layout.addRow("Remember Settings:", self.remember_check)
        
        gui_group.setLayout(gui_layout)
        layout.addWidget(gui_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(browse_button)
        
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "General")
    
    def create_models_tab(self):
        """Create the models settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Detector settings
        detector_group = QGroupBox("Detector Settings")
        detector_layout = QFormLayout()
        
        # Detector type
        self.detector_type_combo = QComboBox()
        self.detector_type_combo.addItems(["yolov8", "tensorflow", "coreml"])
        detector_layout.addRow("Detector Type:", self.detector_type_combo)
        
        # Model name/path
        self.detector_model_edit = QLineEdit()
        detector_layout.addRow("Model Name:", self.detector_model_edit)
        
        # Confidence threshold
        self.detector_confidence_spin = QDoubleSpinBox()
        self.detector_confidence_spin.setRange(0.1, 0.99)
        self.detector_confidence_spin.setSingleStep(0.05)
        self.detector_confidence_spin.setDecimals(2)
        detector_layout.addRow("Confidence Threshold:", self.detector_confidence_spin)
        
        detector_group.setLayout(detector_layout)
        layout.addWidget(detector_group)
        
        # Segmentor settings
        segmentor_group = QGroupBox("Segmentor Settings")
        segmentor_layout = QFormLayout()
        
        # Segmentor type
        self.segmentor_type_combo = QComboBox()
        self.segmentor_type_combo.addItems(["sam", "simple"])
        segmentor_layout.addRow("Segmentor Type:", self.segmentor_type_combo)
        
        # SAM model type
        self.sam_model_type_combo = QComboBox()
        self.sam_model_type_combo.addItems(["vit_h", "vit_l", "vit_b"])
        segmentor_layout.addRow("SAM Model Type:", self.sam_model_type_combo)
        
        # Checkpoint path
        self.sam_checkpoint_edit = QLineEdit()
        segmentor_layout.addRow("SAM Checkpoint:", self.sam_checkpoint_edit)
        
        segmentor_group.setLayout(segmentor_layout)
        layout.addWidget(segmentor_group)
        
        # Tracker settings
        tracker_group = QGroupBox("Tracker Settings")
        tracker_layout = QFormLayout()
        
        # Tracker type
        self.tracker_type_combo = QComboBox()
        self.tracker_type_combo.addItems(["bytetrack", "sort", "simple"])
        tracker_layout.addRow("Tracker Type:", self.tracker_type_combo)
        
        # Track threshold
        self.track_thresh_spin = QDoubleSpinBox()
        self.track_thresh_spin.setRange(0.1, 0.99)
        self.track_thresh_spin.setSingleStep(0.05)
        self.track_thresh_spin.setDecimals(2)
        tracker_layout.addRow("Track Threshold:", self.track_thresh_spin)
        
        # Track buffer
        self.track_buffer_spin = QSpinBox()
        self.track_buffer_spin.setRange(1, 100)
        tracker_layout.addRow("Track Buffer:", self.track_buffer_spin)
        
        # Match threshold
        self.match_thresh_spin = QDoubleSpinBox()
        self.match_thresh_spin.setRange(0.1, 0.99)
        self.match_thresh_spin.setSingleStep(0.05)
        self.match_thresh_spin.setDecimals(2)
        tracker_layout.addRow("Match Threshold:", self.match_thresh_spin)
        
        tracker_group.setLayout(tracker_layout)
        layout.addWidget(tracker_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Models")
    
    def create_llm_tab(self):
        """Create the LLM settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # LLM settings
        llm_group = QGroupBox("LLM Settings")
        llm_layout = QFormLayout()
        
        # LLM type
        self.llm_type_combo = QComboBox()
        self.llm_type_combo.addItems(["none", "ollama", "cloud", "command"])
        self.llm_type_combo.currentTextChanged.connect(self.update_llm_settings_visibility)
        llm_layout.addRow("LLM Type:", self.llm_type_combo)
        
        # Ollama settings
        self.ollama_group = QGroupBox("Ollama Settings")
        ollama_layout = QFormLayout()
        
        self.ollama_model_edit = QLineEdit()
        ollama_layout.addRow("Model:", self.ollama_model_edit)
        
        self.ollama_host_edit = QLineEdit()
        ollama_layout.addRow("Host:", self.ollama_host_edit)
        
        self.ollama_group.setLayout(ollama_layout)
        
        # Cloud API settings
        self.cloud_group = QGroupBox("Cloud API Settings")
        cloud_layout = QFormLayout()
        
        self.cloud_api_key_edit = QLineEdit()
        self.cloud_api_key_edit.setEchoMode(QLineEdit.Password)
        cloud_layout.addRow("API Key:", self.cloud_api_key_edit)
        
        self.cloud_api_endpoint_edit = QLineEdit()
        cloud_layout.addRow("API Endpoint:", self.cloud_api_endpoint_edit)
        
        self.cloud_model_edit = QLineEdit()
        cloud_layout.addRow("Model:", self.cloud_model_edit)
        
        self.cloud_group.setLayout(cloud_layout)
        
        # Command settings
        self.command_group = QGroupBox("Command Settings")
        command_layout = QFormLayout()
        
        self.command_edit = QLineEdit()
        command_layout.addRow("Command:", self.command_edit)
        
        self.command_group.setLayout(command_layout)
        
        # Add all groups to the layout
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        layout.addWidget(self.ollama_group)
        layout.addWidget(self.cloud_group)
        layout.addWidget(self.command_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "LLM")
    
    def create_processing_tab(self):
        """Create the processing settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Processing settings
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout()
        
        # Max objects
        self.max_objects_spin = QSpinBox()
        self.max_objects_spin.setRange(1, 100)
        processing_layout.addRow("Max Objects:", self.max_objects_spin)
        
        # Save results
        self.save_results_check = QCheckBox()
        processing_layout.addRow("Save Results:", self.save_results_check)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16)
        processing_layout.addRow("Batch Size:", self.batch_size_spin)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Processing")
    
    def create_system_tab(self):
        """Create the system settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # System settings
        system_group = QGroupBox("System Settings")
        system_layout = QFormLayout()
        
        # Use MPS (for M1/M2 Macs)
        self.use_mps_check = QCheckBox()
        system_layout.addRow("Use MPS (M1/M2 Macs):", self.use_mps_check)
        
        # Low memory mode
        self.low_memory_mode_check = QCheckBox()
        system_layout.addRow("Low Memory Mode:", self.low_memory_mode_check)
        
        # Debug mode
        self.debug_check = QCheckBox()
        system_layout.addRow("Debug Mode:", self.debug_check)
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["debug", "info", "warning", "error", "critical"])
        system_layout.addRow("Log Level:", self.log_level_combo)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        # Plugin settings
        plugin_group = QGroupBox("Plugin Settings")
        plugin_layout = QFormLayout()
        
        # Plugin directory
        plugin_dir_layout = QHBoxLayout()
        self.plugin_dir_edit = QLineEdit()
        plugin_dir_layout.addWidget(self.plugin_dir_edit)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_plugin_dir)
        plugin_dir_layout.addWidget(browse_button)
        
        plugin_layout.addRow("Plugin Directory:", plugin_dir_layout)
        
        # Enable plugins
        self.enable_plugins_check = QCheckBox()
        plugin_layout.addRow("Enable Plugins:", self.enable_plugins_check)
        
        # Autoload plugins
        self.autoload_plugins_check = QCheckBox()
        plugin_layout.addRow("Autoload Plugins:", self.autoload_plugins_check)
        
        plugin_group.setLayout(plugin_layout)
        layout.addWidget(plugin_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "System")
    
    def update_llm_settings_visibility(self, llm_type):
        """Update visibility of LLM settings based on the selected type."""
        # Hide all LLM settings groups
        self.ollama_group.setVisible(False)
        self.cloud_group.setVisible(False)
        self.command_group.setVisible(False)
        
        # Show the appropriate group
        if llm_type == "ollama":
            self.ollama_group.setVisible(True)
        elif llm_type == "cloud":
            self.cloud_group.setVisible(True)
        elif llm_type == "command":
            self.command_group.setVisible(True)
    
    def load_settings(self):
        """Load settings from the configuration."""
        # GUI settings
        gui_config = self.config.get('gui', {})
        self.width_spin.setValue(gui_config.get('width', 1200))
        self.height_spin.setValue(gui_config.get('height', 800))
        
        theme_idx = self.theme_combo.findText(gui_config.get('theme', 'system'))
        if theme_idx >= 0:
            self.theme_combo.setCurrentIndex(theme_idx)
        
        self.font_size_spin.setValue(gui_config.get('font_size', 10))
        self.remember_check.setChecked(gui_config.get('remember_settings', True))
        
        # Processing settings
        processing_config = self.config.get('processing', {})
        self.output_dir_edit.setText(processing_config.get('output_dir', 'output'))
        self.max_objects_spin.setValue(processing_config.get('max_objects', 20))
        self.save_results_check.setChecked(processing_config.get('save_results', True))
        self.batch_size_spin.setValue(processing_config.get('batch_size', 1))
        
        # Detector settings
        detector_config = self.config.get('models', {}).get('detector', {})
        
        detector_type_idx = self.detector_type_combo.findText(detector_config.get('type', 'yolov8'))
        if detector_type_idx >= 0:
            self.detector_type_combo.setCurrentIndex(detector_type_idx)
        
        self.detector_model_edit.setText(detector_config.get('model_name', 'yolov8n.pt'))
        self.detector_confidence_spin.setValue(detector_config.get('confidence', 0.5))
        
        # Segmentor settings
        segmentor_config = self.config.get('models', {}).get('segmentor', {})
        
        segmentor_type_idx = self.segmentor_type_combo.findText(segmentor_config.get('type', 'sam'))
        if segmentor_type_idx >= 0:
            self.segmentor_type_combo.setCurrentIndex(segmentor_type_idx)
        
        sam_model_type_idx = self.sam_model_type_combo.findText(segmentor_config.get('model_type', 'vit_h'))
        if sam_model_type_idx >= 0:
            self.sam_model_type_combo.setCurrentIndex(sam_model_type_idx)
        
        self.sam_checkpoint_edit.setText(segmentor_config.get('checkpoint', 'sam_vit_h_4b8939.pth'))
        
        # Tracker settings
        tracker_config = self.config.get('models', {}).get('tracker', {})
        
        tracker_type_idx = self.tracker_type_combo.findText(tracker_config.get('type', 'bytetrack'))
        if tracker_type_idx >= 0:
            self.tracker_type_combo.setCurrentIndex(tracker_type_idx)
        
        self.track_thresh_spin.setValue(tracker_config.get('track_thresh', 0.25))
        self.track_buffer_spin.setValue(tracker_config.get('track_buffer', 30))
        self.match_thresh_spin.setValue(tracker_config.get('match_thresh', 0.8))
        
        # LLM settings
        llm_config = self.config.get('llm', {})
        
        llm_type_idx = self.llm_type_combo.findText(llm_config.get('type', 'none'))
        if llm_type_idx >= 0:
            self.llm_type_combo.setCurrentIndex(llm_type_idx)
        
        self.ollama_model_edit.setText(llm_config.get('model', 'llama3'))
        self.ollama_host_edit.setText(llm_config.get('host', 'http://localhost:11434'))
        
        self.cloud_api_key_edit.setText(llm_config.get('api_key', ''))
        self.cloud_api_endpoint_edit.setText(llm_config.get('api_endpoint', 'https://api.openai.com/v1/chat/completions'))
        self.cloud_model_edit.setText(llm_config.get('model', 'gpt-4-vision-preview'))
        
        self.command_edit.setText(llm_config.get('command', 'ollama'))
        
        # System settings
        system_config = self.config.get('system', {})
        self.use_mps_check.setChecked(system_config.get('use_mps', True))
        self.low_memory_mode_check.setChecked(system_config.get('low_memory_mode', False))
        self.debug_check.setChecked(system_config.get('debug', False))
        
        log_level_idx = self.log_level_combo.findText(system_config.get('log_level', 'info'))
        if log_level_idx >= 0:
            self.log_level_combo.setCurrentIndex(log_level_idx)
        
        # Plugin settings
        plugin_config = self.config.get('plugins', {})
        self.plugin_dir_edit.setText(plugin_config.get('directory', 'plugins'))
        self.enable_plugins_check.setChecked(plugin_config.get('enabled', True))
        self.autoload_plugins_check.setChecked(plugin_config.get('autoload', True))
        
        # Update LLM settings visibility
        self.update_llm_settings_visibility(self.llm_type_combo.currentText())
    
    def save_settings(self):
        """Save settings to the configuration."""
        try:
            # Create configuration structure
            self.config['gui'] = {
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'theme': self.theme_combo.currentText(),
                'font_size': self.font_size_spin.value(),
                'remember_settings': self.remember_check.isChecked()
            }
            
            self.config['processing'] = {
                'output_dir': self.output_dir_edit.text(),
                'max_objects': self.max_objects_spin.value(),
                'save_results': self.save_results_check.isChecked(),
                'batch_size': self.batch_size_spin.value()
            }
            
            # Ensure models configuration exists
            if 'models' not in self.config:
                self.config['models'] = {}
            
            self.config['models']['detector'] = {
                'type': self.detector_type_combo.currentText(),
                'model_name': self.detector_model_edit.text(),
                'confidence': self.detector_confidence_spin.value()
            }
            
            self.config['models']['segmentor'] = {
                'type': self.segmentor_type_combo.currentText(),
                'model_type': self.sam_model_type_combo.currentText(),
                'checkpoint': self.sam_checkpoint_edit.text()
            }
            
            self.config['models']['tracker'] = {
                'type': self.tracker_type_combo.currentText(),
                'track_thresh': self.track_thresh_spin.value(),
                'track_buffer': self.track_buffer_spin.value(),
                'match_thresh': self.match_thresh_spin.value()
            }
            
            # LLM configuration
            llm_type = self.llm_type_combo.currentText()
            
            self.config['llm'] = {
                'type': llm_type
            }
            
            # Add type-specific LLM settings
            if llm_type == 'ollama':
                self.config['llm']['model'] = self.ollama_model_edit.text()
                self.config['llm']['host'] = self.ollama_host_edit.text()
            elif llm_type == 'cloud':
                self.config['llm']['api_key'] = self.cloud_api_key_edit.text()
                self.config['llm']['api_endpoint'] = self.cloud_api_endpoint_edit.text()
                self.config['llm']['model'] = self.cloud_model_edit.text()
            elif llm_type == 'command':
                self.config['llm']['command'] = self.command_edit.text()
            
            # System configuration
            self.config['system'] = {
                'use_mps': self.use_mps_check.isChecked(),
                'low_memory_mode': self.low_memory_mode_check.isChecked(),
                'debug': self.debug_check.isChecked(),
                'log_level': self.log_level_combo.currentText()
            }
            
            # Plugin configuration
            self.config['plugins'] = {
                'directory': self.plugin_dir_edit.text(),
                'enabled': self.enable_plugins_check.isChecked(),
                'autoload': self.autoload_plugins_check.isChecked()
            }
            
            # Accept the dialog
            self.accept()
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"Error saving settings: {str(e)}")
    
    def get_config(self):
        """Get the updated configuration."""
        return self.config
    
    def export_settings(self):
        """Export settings to a YAML file."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("YAML files (*.yaml *.yml)")
        file_dialog.setDefaultSuffix("yaml")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                # Save current settings to config
                self.save_settings()
                
                # Export to file
                with open(file_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
                
                QMessageBox.information(self, "Export Successful", f"Settings exported to {file_path}")
                
            except Exception as e:
                logger.error(f"Error exporting settings: {e}")
                QMessageBox.critical(self, "Error", f"Error exporting settings: {str(e)}")
    
    def import_settings(self):
        """Import settings from a YAML file."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("YAML files (*.yaml *.yml)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                # Load from file
                with open(file_path, 'r') as f:
                    imported_config = yaml.safe_load(f)
                
                # Update configuration
                self.config = imported_config
                
                # Reload UI
                self.load_settings()
                
                QMessageBox.information(self, "Import Successful", f"Settings imported from {file_path}")
                
            except Exception as e:
                logger.error(f"Error importing settings: {e}")
                QMessageBox.critical(self, "Error", f"Error importing settings: {str(e)}")
    
    def reset_to_defaults(self):
        """Reset settings to defaults."""
        # Confirm with user
        result = QMessageBox.question(
            self, 
            "Confirm Reset", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            try:
                # Load default config
                with open('config.yaml', 'r') as f:
                    self.config = yaml.safe_load(f)
                
                # Reload UI
                self.load_settings()
                
                QMessageBox.information(self, "Reset Successful", "Settings reset to defaults")
                
            except Exception as e:
                logger.error(f"Error resetting settings: {e}")
                QMessageBox.critical(self, "Error", f"Error resetting settings: {str(e)}")
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        
        if dir_dialog.exec_():
            dir_path = dir_dialog.selectedFiles()[0]
            self.output_dir_edit.setText(dir_path)
    
    def browse_plugin_dir(self):
        """Browse for plugin directory."""
        dir_dialog = QFileDialog()
        dir_dialog.setFileMode(QFileDialog.Directory)
        
        if dir_dialog.exec_():
            dir_path = dir_dialog.selectedFiles()[0]
            self.plugin_dir_edit.setText(dir_path)
