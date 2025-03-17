#!/usr/bin/env python3
import argparse
import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('object_detection')

# Add module paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth.login import LoginWindow
from gui.main_window import MainWindow
from utils.system_info import detect_system
from utils.preprocessing import check_dependencies


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Object Detection and Tracking System')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run in command-line mode without GUI')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--skip-login', action='store_true',
                        help='Skip login screen (development only)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return {}


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Detect system capabilities
    system_info = detect_system()
    logger.info(f"System detected: {system_info['os']} on {system_info['architecture']}")
    logger.info(f"Hardware acceleration: {system_info['gpu_info']}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply system-specific optimizations
    if system_info['os'] == 'Darwin' and 'arm' in system_info['architecture']:
        # M1/M2 Mac specific optimizations
        logger.info("Applying Apple Silicon optimizations")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        config.setdefault('system', {})['use_mps'] = True
    
    # Check dependencies and GPU availability
    check_dependencies(system_info)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Start with login window if not skipped
    if args.skip_login:
        logger.info("Login skipped")
        window = MainWindow(args, config, system_info)
        window.show()
    else:
        login = LoginWindow(config)
        if login.exec_():
            # Login successful
            window = MainWindow(args, config, system_info, user=login.get_user())
            window.show()
        else:
            # Login canceled or failed
            logger.info("Login canceled or failed")
            return 1
    
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
