import platform
import os
import sys
import logging
import subprocess
import psutil
import torch

logger = logging.getLogger('object_detection.system_info')

def detect_system():
    """
    Detect system information including OS, architecture, GPU, and memory.
    
    Returns:
        Dictionary containing system information
    """
    system_info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=True),
        'physical_cpu_count': psutil.cpu_count(logical=False),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'gpu_available': False,
        'gpu_info': 'None',
        'acceleration': 'CPU'
    }
    
    # Detect GPU capabilities
    try:
        # Check for CUDA
        if torch.cuda.is_available():
            system_info['gpu_available'] = True
            system_info['gpu_info'] = f"CUDA {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}"
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
            system_info['acceleration'] = 'CUDA'
        
        # Check for MPS (Apple Metal)
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            system_info['gpu_available'] = True
            system_info['gpu_info'] = "Apple Metal Performance Shaders (MPS)"
            system_info['acceleration'] = 'MPS'
        
        # Additional GPU detection for macOS
        elif system_info['os'] == 'Darwin':
            try:
                # Try to get GPU info on macOS
                gpu_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], 
                                                 text=True)
                if "Apple M" in gpu_info:
                    # This is an Apple Silicon Mac
                    system_info['gpu_info'] = "Apple Silicon GPU"
                    # Even if PyTorch MPS is not available, we can still use Core ML
                    system_info['acceleration'] = 'CoreML'
            except:
                pass
    
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")
    
    logger.info(f"System detection complete: {system_info}")
    return system_info

def optimize_for_system(system_info):
    """
    Apply system-specific optimizations.
    
    Args:
        system_info: Dictionary containing system information
    """
    # M1/M2 Mac specific optimizations
    if system_info['os'] == 'Darwin' and 'arm' in system_info['architecture']:
        logger.info("Applying Apple Silicon optimizations")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Check for Ollama
        try:
            ollama_present = subprocess.run(['which', 'ollama'], 
                                         stdout=subprocess.PIPE).returncode == 0
            if ollama_present:
                logger.info("Ollama detected on system")
                system_info['ollama_available'] = True
        except:
            system_info['ollama_available'] = False
    
    # CUDA specific optimizations
    if 'CUDA' in system_info.get('acceleration', ''):
        logger.info("Applying CUDA optimizations")
        # Set appropriate CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        
    # Memory optimizations based on available RAM
    available_gb = system_info['memory_available'] / (1024**3)
    if available_gb < 4:
        logger.info("Low memory mode enabled")
        system_info['low_memory_mode'] = True
    else:
        system_info['low_memory_mode'] = False
        
    return system_info

def get_optimal_batch_size(system_info):
    """
    Determine the optimal batch size based on the available hardware.
    
    Args:
        system_info: Dictionary containing system information
        
    Returns:
        Recommended batch size
    """
    # Default for CPU
    batch_size = 1
    
    # If CUDA is available, adjust based on GPU memory
    if 'CUDA' in system_info.get('acceleration', ''):
        gpu_memory_gb = system_info.get('gpu_memory', 0) / (1024**3)
        if gpu_memory_gb > 16:
            batch_size = 8
        elif gpu_memory_gb > 8:
            batch_size = 4
        elif gpu_memory_gb > 4:
            batch_size = 2
    
    # For Apple Silicon, use smaller batch sizes initially
    elif 'MPS' in system_info.get('acceleration', '') or 'CoreML' in system_info.get('acceleration', ''):
        available_memory_gb = system_info['memory_available'] / (1024**3)
        if available_memory_gb > 24:  # High-end M1 Max/M2 Max
            batch_size = 4
        elif available_memory_gb > 16:  # Mid-range M1/M2
            batch_size = 2
    
    # Adjust for low memory mode
    if system_info.get('low_memory_mode', False):
        batch_size = max(1, batch_size // 2)
    
    logger.info(f"Optimal batch size determined: {batch_size}")
    return batch_size
