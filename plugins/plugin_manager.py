import os
import sys
import importlib.util
import logging
import inspect
from models.base import BasePlugin

logger = logging.getLogger('object_detection.plugins')

class PluginManager:
    """Class for managing plugins."""
    
    def __init__(self, config=None):
        """
        Initialize the plugin manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.plugins_dir = self.config.get('plugins', {}).get('directory', 'plugins')
        self.plugins = {}
        
        # Load plugins
        self.load_plugins()
    
    def load_plugins(self):
        """Load plugins from the plugins directory."""
        # Clear existing plugins
        self.plugins = {}
        
        # Ensure plugins directory exists
        plugins_dir = os.path.abspath(self.plugins_dir)
        if not os.path.exists(plugins_dir):
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return
        
        # Add plugins directory to path if not already
        if plugins_dir not in sys.path:
            sys.path.append(plugins_dir)
        
        # Find plugin files
        plugin_files = []
        for root, dirs, files in os.walk(plugins_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('_'):
                    plugin_files.append(os.path.join(root, file))
        
        # Load each plugin
        for plugin_file in plugin_files:
            try:
                self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file}: {e}")
    
    def _load_plugin_from_file(self, plugin_file):
        """
        Load a plugin from a file.
        
        Args:
            plugin_file: Path to the plugin file
        """
        try:
            # Get module name from file path
            module_name = os.path.splitext(os.path.basename(plugin_file))[0]
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, BasePlugin) and 
                    obj is not BasePlugin):
                    
                    # Create instance of the plugin
                    plugin = obj(self.config)
                    
                    # Get plugin info
                    info = plugin.get_info()
                    
                    # Store the plugin
                    plugin_id = info.get('id', name)
                    self.plugins[plugin_id] = plugin
                    
                    logger.info(f"Loaded plugin: {plugin_id} - {info.get('name', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_file}: {e}")
            raise
    
    def get_plugin(self, plugin_id):
        """
        Get a plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_all_plugins(self):
        """
        Get all loaded plugins.
        
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return self.plugins
    
    def get_plugins_by_type(self, plugin_type):
        """
        Get plugins by type.
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return {
            plugin_id: plugin
            for plugin_id, plugin in self.plugins.items()
            if plugin.get_info().get('type') == plugin_type
        }
    
    def process_with_plugin(self, plugin_id, data):
        """
        Process data with a plugin.
        
        Args:
            plugin_id: Plugin ID
            data: Input data
            
        Returns:
            Processed data or None if plugin not found
        """
        plugin = self.get_plugin(plugin_id)
        
        if plugin:
            try:
                return plugin.process(data)
            except Exception as e:
                logger.error(f"Error processing with plugin {plugin_id}: {e}")
                return None
        else:
            logger.warning(f"Plugin not found: {plugin_id}")
            return None


# Example plugin template
class ExamplePlugin(BasePlugin):
    """Example plugin class."""
    
    def get_info(self):
        """
        Get plugin information.
        
        Returns:
            Dictionary containing plugin information
        """
        return {
            'id': 'example_plugin',
            'name': 'Example Plugin',
            'description': 'This is an example plugin',
            'version': '1.0',
            'author': 'Your Name',
            'type': 'example'
        }
    
    def process(self, data):
        """
        Process data.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        # Example processing
        return data
