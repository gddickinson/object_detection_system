import os
import cv2
import numpy as np
import logging
from PIL import Image
import json

logger = logging.getLogger('object_detection.utils.format_converter')

def read_image(file_path):
    """
    Read an image file in various formats.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image as a numpy array in BGR format, or None if reading fails
    """
    try:
        # Check file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            # Standard image formats - use OpenCV
            image = cv2.imread(file_path)
            return image
        
        elif ext in ['.webp']:
            # WebP format - use PIL and convert to OpenCV format
            pil_image = Image.open(file_path)
            rgb_image = np.array(pil_image.convert('RGB'))
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            return bgr_image
        
        elif ext in ['.heic', '.heif']:
            # HEIC/HEIF format - use PIL with special handling
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                
                pil_image = Image.open(file_path)
                rgb_image = np.array(pil_image.convert('RGB'))
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                return bgr_image
            except ImportError:
                logger.error("pillow_heif package is required for HEIC/HEIF format. Install with 'pip install pillow-heif'")
                return None
        
        else:
            # Try to read with OpenCV as a fallback
            image = cv2.imread(file_path)
            if image is None:
                # If OpenCV fails, try PIL
                try:
                    pil_image = Image.open(file_path)
                    rgb_image = np.array(pil_image.convert('RGB'))
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    return bgr_image
                except:
                    logger.error(f"Unsupported image format: {ext}")
                    return None
            else:
                return image
    
    except Exception as e:
        logger.error(f"Error reading image {file_path}: {e}")
        return None


def is_video_file(file_path):
    """
    Check if a file is a video.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a video, False otherwise
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    return ext in video_extensions


def is_image_file(file_path):
    """
    Check if a file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an image, False otherwise
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    image_extensions = [
        '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', 
        '.webp', '.heic', '.heif', '.gif'
    ]
    
    return ext in image_extensions


def read_video_info(file_path):
    """
    Read video file information.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary containing video information, or None if reading fails
    """
    try:
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {file_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Release the video capture
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count
        }
    
    except Exception as e:
        logger.error(f"Error reading video info {file_path}: {e}")
        return None


def save_image(image, file_path, quality=95):
    """
    Save an image to a file.
    
    Args:
        image: Image as a numpy array in BGR format
        file_path: Path to save the image to
        quality: JPEG quality (0-100)
        
    Returns:
        True if saving succeeded, False otherwise
    """
    try:
        # Check file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        if ext in ['.jpg', '.jpeg']:
            # JPEG format
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            return cv2.imwrite(file_path, image, params)
        
        elif ext in ['.png']:
            # PNG format
            params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)]
            return cv2.imwrite(file_path, image, params)
        
        elif ext in ['.webp']:
            # WebP format
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            return cv2.imwrite(file_path, image, params)
        
        else:
            # Default save
            return cv2.imwrite(file_path, image)
    
    except Exception as e:
        logger.error(f"Error saving image {file_path}: {e}")
        return False


def convert_image_format(input_path, output_path, quality=95):
    """
    Convert an image from one format to another.
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the output image
        quality: Output quality (0-100)
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Read input image
        image = read_image(input_path)
        
        if image is None:
            return False
        
        # Save to output format
        return save_image(image, output_path, quality)
    
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        return False


def convert_video_format(input_path, output_path, output_format='mp4', quality=95):
    """
    Convert a video from one format to another.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        output_format: Output format (e.g., 'mp4', 'avi')
        quality: Output quality (0-100)
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Open input video
        input_cap = cv2.VideoCapture(input_path)
        
        if not input_cap.isOpened():
            logger.error(f"Failed to open input video: {input_path}")
            return False
        
        # Get video properties
        width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = input_cap.get(cv2.CAP_PROP_FPS)
        
        # Set output video properties based on format
        if output_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_format == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            # Default to MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create output video writer
        output_cap = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not output_cap.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            input_cap.release()
            return False
        
        # Process frames
        while True:
            ret, frame = input_cap.read()
            
            if not ret:
                break
            
            # Write frame to output
            output_cap.write(frame)
        
        # Release video objects
        input_cap.release()
        output_cap.release()
        
        return True
    
    except Exception as e:
        logger.error(f"Error converting video: {e}")
        return False


def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None):
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        frame_interval: Interval between extracted frames (1 = every frame)
        max_frames: Maximum number of frames to extract (None = all frames)
        
    Returns:
        Number of extracted frames, or -1 if extraction fails
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return -1
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate number of frames to extract
        if max_frames is None:
            max_frames = frame_count
        
        # Extract frames
        frame_idx = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at interval
            if frame_idx % frame_interval == 0:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                extracted_count += 1
                
                # Check if we've reached the maximum
                if extracted_count >= max_frames:
                    break
            
            frame_idx += 1
        
        # Release video object
        cap.release()
        
        return extracted_count
    
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return -1


def create_video_from_images(image_dir, output_path, fps=30, image_pattern="*.jpg", sort_by_name=True):
    """
    Create a video from a series of images.
    
    Args:
        image_dir: Directory containing the images
        output_path: Path to save the output video
        fps: Frames per second
        image_pattern: Pattern to match image files
        sort_by_name: Whether to sort images by name
        
    Returns:
        True if creation succeeded, False otherwise
    """
    try:
        import glob
        
        # Get list of images
        image_paths = glob.glob(os.path.join(image_dir, image_pattern))
        
        if not image_paths:
            logger.error(f"No images found in {image_dir} matching pattern {image_pattern}")
            return False
        
        # Sort images if needed
        if sort_by_name:
            image_paths.sort()
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_paths[0])
        height, width, _ = first_image.shape
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            return False
        
        # Process images
        for image_path in image_paths:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            
            # Resize if dimensions don't match
            if image.shape[0] != height or image.shape[1] != width:
                image = cv2.resize(image, (width, height))
            
            # Write to video
            video_writer.write(image)
        
        # Release video writer
        video_writer.release()
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating video from images: {e}")
        return False


def save_results_to_json(results, output_path):
    """
    Save detection/tracking results to a JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save the JSON file
        
    Returns:
        True if saving succeeded, False otherwise
    """
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        
        if 'detections' in results:
            json_results['detections'] = {
                'boxes': results['detections']['boxes'].tolist() if len(results['detections']['boxes']) > 0 else [],
                'scores': results['detections']['scores'].tolist() if len(results['detections']['scores']) > 0 else [],
                'classes': results['detections']['classes'].tolist() if len(results['detections']['classes']) > 0 else [],
                'class_names': results['detections']['class_names'],
                'centers': results['detections']['centers'].tolist() if len(results['detections']['centers']) > 0 else []
            }
            
            if 'llm_descriptions' in results['detections']:
                json_results['detections']['llm_descriptions'] = results['detections']['llm_descriptions']
        
        if 'tracking' in results:
            # Convert track data
            tracks_json = {}
            
            for track_id, track in results['tracking']['tracks'].items():
                tracks_json[str(track_id)] = {
                    'boxes': [box.tolist() if hasattr(box, 'tolist') else box for box in track['boxes']],
                    'class': track.get('class', 0),
                    'first_frame': track.get('first_frame', 0),
                    'last_frame': track.get('last_frame', 0)
                }
            
            json_results['tracking'] = {
                'tracks': tracks_json,
                'active_tracks': results['tracking']['active_tracks']
            }
        
        if 'frame_index' in results:
            json_results['frame_index'] = results['frame_index']
        
        if 'total_frames' in results:
            json_results['total_frames'] = results['total_frames']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        return False


def load_results_from_json(json_path):
    """
    Load detection/tracking results from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Results dictionary or None if loading fails
    """
    try:
        # Load from file
        with open(json_path, 'r') as f:
            json_results = json.load(f)
        
        # Convert lists to numpy arrays
        results = {}
        
        if 'detections' in json_results:
            results['detections'] = {
                'boxes': np.array(json_results['detections']['boxes']),
                'scores': np.array(json_results['detections']['scores']),
                'classes': np.array(json_results['detections']['classes']),
                'class_names': json_results['detections']['class_names'],
                'centers': np.array(json_results['detections']['centers'])
            }
            
            if 'llm_descriptions' in json_results['detections']:
                results['detections']['llm_descriptions'] = json_results['detections']['llm_descriptions']
        
        if 'tracking' in json_results:
            # Convert track data
            tracks = {}
            
            for track_id, track in json_results['tracking']['tracks'].items():
                tracks[int(track_id)] = {
                    'boxes': [np.array(box) if isinstance(box, list) else box for box in track['boxes']],
                    'class': track.get('class', 0),
                    'first_frame': track.get('first_frame', 0),
                    'last_frame': track.get('last_frame', 0)
                }
            
            results['tracking'] = {
                'tracks': tracks,
                'active_tracks': json_results['tracking']['active_tracks']
            }
        
        if 'frame_index' in json_results:
            results['frame_index'] = json_results['frame_index']
        
        if 'total_frames' in json_results:
            results['total_frames'] = json_results['total_frames']
        
        return results
    
    except Exception as e:
        logger.error(f"Error loading results from JSON: {e}")
        return None
