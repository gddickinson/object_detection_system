import sys
import subprocess
import importlib
import logging
import platform
import pkg_resources
import os

logger = logging.getLogger('object_detection.utils.preprocessing')

def check_dependencies(system_info=None):
    """
    Check if required packages are installed and hardware is available.
    
    Args:
        system_info: Dictionary containing system information
    """
    # Required packages
    required_packages = [
        'torch',
        'numpy',
        'opencv-python',
        'PyQt5',
        'pyyaml',
        'psutil'
    ]
    
    # Optional packages
    optional_packages = [
        'ultralytics',  # For YOLOv8
        'segment_anything',  # For SAM
        'coremltools',  # For CoreML support on macOS
        'tensorflow',  # For TensorFlow models
        'onnxruntime',  # For ONNX models
        'requests',  # For API calls
        'Pillow',  # For image processing
    ]
    
    # Set of missing required packages
    missing_required = []
    
    # Set of missing optional packages
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        if not is_package_installed(package):
            missing_required.append(package)
    
    # Log warning for missing required packages
    if missing_required:
        logger.warning(f"Missing required packages: {', '.join(missing_required)}")
        logger.warning("Please install them using pip: pip install " + " ".join(missing_required))
    
    # Check optional packages
    for package in optional_packages:
        if not is_package_installed(package):
            missing_optional.append(package)
    
    # Log info for missing optional packages
    if missing_optional:
        logger.info(f"Missing optional packages: {', '.join(missing_optional)}")
        logger.info("Some features may not be available.")
    
    # Check GPU availability
    check_gpu_availability(system_info)
    
    # Check for specific packages based on OS
    if platform.system() == 'Darwin':
        # Check for CoreML on macOS
        if 'coremltools' not in missing_optional:
            logger.info("CoreML support is available")
        
        # Check for MPS (Metal Performance Shaders) on Apple Silicon
        if system_info and system_info.get('architecture') == 'arm64':
            check_mps_availability()
    
    # Print final status
    if not missing_required:
        logger.info("All required dependencies are installed")
    else:
        logger.warning("Some required dependencies are missing")


def is_package_installed(package_name):
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if the package is installed, False otherwise
    """
    try:
        # Try to get package version
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def install_package(package_name):
    """
    Install a package using pip.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        True if the installation was successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def check_gpu_availability(system_info=None):
    """
    Check if a GPU is available for computation.
    
    Args:
        system_info: Dictionary containing system information
    """
    try:
        # Try to import torch
        import torch
        
        # Check for CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            
            logger.info(f"CUDA is available: {cuda_version}, Device: {device_name}")
            return True
        
        # Check for MPS (Apple Metal)
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available")
            return True
        
        else:
            logger.warning("No GPU acceleration available, using CPU")
            return False
    
    except ImportError:
        logger.warning("PyTorch is not installed, cannot check GPU availability")
        return False


def check_mps_availability():
    """
    Check if MPS (Metal Performance Shaders) is available on macOS.
    """
    try:
        # Try to import torch
        import torch
        
        # Check for MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available")
            
            # Set environment variable for MPS fallback
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            return True
        else:
            logger.warning("MPS is not available")
            return False
    
    except ImportError:
        logger.warning("PyTorch is not installed, cannot check MPS availability")
        return False


def check_ollama_availability():
    """
    Check if Ollama is available on the system.
    
    Returns:
        Dictionary with Ollama availability info or None if not available
    """
    try:
        # Check if Ollama is installed
        ollama_path = subprocess.run(["which", "ollama"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        
        if ollama_path.returncode != 0:
            logger.info("Ollama is not installed or not in PATH")
            return None
        
        ollama_path = ollama_path.stdout.strip()
        
        # Check if Ollama service is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            
            if response.status_code == 200:
                # Get available models
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                logger.info(f"Ollama is running with models: {available_models}")
                
                return {
                    'available': True,
                    'path': ollama_path,
                    'models': available_models
                }
            else:
                logger.info("Ollama service is not running or not responding correctly")
                return {
                    'available': False,
                    'path': ollama_path,
                    'error': f"HTTP status {response.status_code}"
                }
                
        except Exception as e:
            logger.info(f"Could not connect to Ollama service: {e}")
            return {
                'available': False,
                'path': ollama_path,
                'error': str(e)
            }
    
    except Exception as e:
        logger.info(f"Error checking for Ollama: {e}")
        return None


def download_model_if_needed(model_path, url, size_hint=None):
    """
    Download a model file if it doesn't exist.
    
    Args:
        model_path: Path where the model should be stored
        url: URL to download the model from
        size_hint: Hint about the file size for progress reporting
        
    Returns:
        True if the model is available, False otherwise
    """
    # If model already exists, return True
    if os.path.exists(model_path):
        return True
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to import requests
        import requests
        from tqdm import tqdm
        
        logger.info(f"Downloading model from {url} to {model_path}")
        
        # Start the download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        if size_hint and total_size == 0:
            total_size = size_hint
        
        # Progress bar
        print(f"Downloading {os.path.basename(model_path)}...")
        
        with open(model_path, 'wb') as f, tqdm(
            desc=os.path.basename(model_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))
        
        logger.info(f"Model downloaded to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False
