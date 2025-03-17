"""Utility functions for the object detection application."""

from utils.preprocessing import check_dependencies, download_model_if_needed
from utils.system_info import detect_system, optimize_for_system, get_optimal_batch_size
from utils.format_converter import (
    read_image, is_video_file, is_image_file, read_video_info,
    save_image, convert_image_format, convert_video_format,
    extract_frames, create_video_from_images,
    save_results_to_json, load_results_from_json
)
