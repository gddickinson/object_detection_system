import requests
import base64
import json
import numpy as np
import cv2
from PIL import Image
import io
import logging
import subprocess
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger('object_detection.models.llm')

class BaseLLM(ABC):
    """Abstract base class for LLM integration."""
    
    @abstractmethod
    def analyze_image(self, image):
        """
        Analyze an image using LLM.
        
        Args:
            image: Image to analyze
            
        Returns:
            Description of the image
        """
        pass


class CloudLLM(BaseLLM):
    """Class for cloud-based LLM integration (e.g., OpenAI, Anthropic)."""
    
    def __init__(self, api_key=None, api_endpoint=None, model="gpt-4-vision-preview"):
        """
        Initialize the cloud LLM analyzer.
        
        Args:
            api_key: API key for the LLM service
            api_endpoint: API endpoint for the LLM service
            model: Model name
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model = model
        
        if not api_key or not api_endpoint:
            logger.warning("API key or endpoint not provided for Cloud LLM")
    
    def analyze_image(self, image):
        """
        Analyze an image using cloud LLM.
        
        Args:
            image: Image to analyze
            
        Returns:
            Description of the image
        """
        if not self.api_key or not self.api_endpoint:
            return "Unable to analyze image (API key or endpoint not provided)"
        
        try:
            # Convert image to bytes
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Encode image to base64
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What object is this? Provide a brief description in 5 words or less."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 20
            }
            
            # Make request
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=10)
            result = response.json()
            
            # Extract description
            try:
                description = result["choices"][0]["message"]["content"].strip()
                return description
            except:
                logger.error(f"Error extracting description from response: {result}")
                return "Unidentified object"
        
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return "Error analyzing image"


class OllamaLLM(BaseLLM):
    """Class for local Ollama LLM integration."""
    
    def __init__(self, model="llama3", host="http://localhost:11434"):
        """
        Initialize the Ollama LLM analyzer.
        
        Args:
            model: Ollama model name
            host: Ollama host address
        """
        self.model = model
        self.host = host
        self.api_endpoint = f"{host}/api/generate"
        
        # Check if Ollama is available
        self.available = self._check_availability()
        
        if not self.available:
            logger.warning("Ollama is not available")
    
    def _check_availability(self):
        """
        Check if Ollama is available.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            
            if response.status_code == 200:
                # Check if the model is available
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if self.model in available_models:
                    logger.info(f"Ollama model {self.model} is available")
                    return True
                else:
                    logger.warning(f"Ollama model {self.model} is not available. Available models: {available_models}")
                    
                    # If our model isn't available, but there are other models, use the first one
                    if available_models:
                        self.model = available_models[0]
                        logger.info(f"Using Ollama model {self.model} instead")
                        return True
                    
                    return False
            else:
                logger.warning(f"Failed to get Ollama models: {response.status_code}")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error connecting to Ollama: {e}")
            return False
    
    def analyze_image(self, image):
        """
        Analyze an image using Ollama.
        
        Args:
            image: Image to analyze
            
        Returns:
            Description of the image
        """
        if not self.available:
            return "Ollama is not available"
        
        try:
            # Save image to temporary file
            temp_path = os.path.join(os.path.dirname(__file__), "temp_img.jpg")
            cv2.imwrite(temp_path, image)
            
            # Prepare prompt
            prompt = "This is an image of an object. Describe the object in 5 words or less."
            
            # Prepare request
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [temp_path],
                "stream": False
            }
            
            # Make request
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=10)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()
                
                # Limit to first 5 words if needed
                words = description.split()
                if len(words) > 5:
                    description = " ".join(words[:5])
                
                return description
            else:
                logger.error(f"Error from Ollama: {response.status_code} - {response.text}")
                return "Error analyzing image"
        
        except Exception as e:
            logger.error(f"Error analyzing image with Ollama: {e}")
            return "Error analyzing image"


class LocalCommandLLM(BaseLLM):
    """Class for local command-line based LLM integration."""
    
    def __init__(self, command="ollama"):
        """
        Initialize the local command LLM analyzer.
        
        Args:
            command: Command to run
        """
        self.command = command
        
        # Check if command is available
        self.available = self._check_availability()
        
        if not self.available:
            logger.warning(f"Command {command} is not available")
    
    def _check_availability(self):
        """
        Check if the command is available.
        
        Returns:
            True if the command is available, False otherwise
        """
        try:
            # Check if command exists
            result = subprocess.run(["which", self.command], capture_output=True, text=True)
            
            return result.returncode == 0
        except:
            return False
    
    def analyze_image(self, image):
        """
        Analyze an image using local command.
        
        Args:
            image: Image to analyze
            
        Returns:
            Description of the image
        """
        if not self.available:
            return f"Command {self.command} is not available"
        
        try:
            # Save image to temporary file
            temp_path = os.path.join(os.path.dirname(__file__), "temp_img.jpg")
            cv2.imwrite(temp_path, image)
            
            # Run command
            cmd = [self.command, "run", "llama3", f"Describe the object in this image in 5 words or less: {temp_path}"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if result.returncode == 0:
                description = result.stdout.strip()
                
                # Limit to first 5 words if needed
                words = description.split()
                if len(words) > 5:
                    description = " ".join(words[:5])
                
                return description
            else:
                logger.error(f"Error from command: {result.stderr}")
                return "Error analyzing image"
        
        except Exception as e:
            logger.error(f"Error analyzing image with command: {e}")
            return "Error analyzing image"


class LLMAnalyzer:
    """Class for integrating LLM capabilities for enhanced object recognition."""
    
    def __init__(self, config=None, system_info=None):
        """
        Initialize the LLM analyzer.
        
        Args:
            config: Configuration dictionary
            system_info: System information dictionary
        """
        self.config = config or {}
        self.system_info = system_info or {}
        
        # LLM settings
        llm_config = self.config.get('llm', {})
        self.llm_type = llm_config.get('type', 'none')
        
        # Initialize LLM
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """
        Initialize the appropriate LLM based on configuration.
        
        Returns:
            LLM instance
        """
        if self.llm_type == 'cloud':
            # Cloud LLM (OpenAI, etc.)
            llm_config = self.config.get('llm', {})
            return CloudLLM(
                api_key=llm_config.get('api_key'),
                api_endpoint=llm_config.get('api_endpoint'),
                model=llm_config.get('model', 'gpt-4-vision-preview')
            )
        
        elif self.llm_type == 'ollama':
            # Ollama LLM
            llm_config = self.config.get('llm', {})
            return OllamaLLM(
                model=llm_config.get('model', 'llama3'),
                host=llm_config.get('host', 'http://localhost:11434')
            )
        
        elif self.llm_type == 'command':
            # Command-line LLM
            llm_config = self.config.get('llm', {})
            return LocalCommandLLM(
                command=llm_config.get('command', 'ollama')
            )
        
        else:
            # No LLM
            return None
    
    def set_llm_type(self, llm_type):
        """
        Set the LLM type.
        
        Args:
            llm_type: LLM type ('cloud', 'ollama', 'command', 'none')
        """
        self.llm_type = llm_type
        self.llm = self._init_llm()
    
    def analyze_objects(self, image, detections):
        """
        Analyze detected objects using an LLM.
        
        Args:
            image: Original image
            detections: Dictionary containing detection results
            
        Returns:
            Enhanced detections with LLM-provided descriptions
        """
        if not self.llm:
            return detections
        
        results = detections.copy()
        descriptions = []
        
        # For each detection, extract the region and analyze it
        for i, box in enumerate(detections['boxes']):
            # Extract object region
            x1, y1, x2, y2 = map(int, box)
            object_img = image[y1:y2, x1:x2]
            
            # Skip if the region is invalid
            if object_img.size == 0:
                descriptions.append(detections['class_names'][i])
                continue
            
            # Get description from LLM
            try:
                description = self.llm.analyze_image(object_img)
                descriptions.append(description)
            except Exception as e:
                logger.error(f"Error getting LLM description: {e}")
                descriptions.append(detections['class_names'][i])
        
        results['llm_descriptions'] = descriptions
        return results
