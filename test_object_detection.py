#!/usr/bin/env python3
"""
Test suite for the Object Detection and Tracking System.
Smoke tests for detector, segmentor, tracker, and utility modules.
"""

import unittest
import os
import sys
import json
import tempfile
import hashlib
from unittest.mock import patch, MagicMock

# Ensure project root is on path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


class TestDetectorValidation(unittest.TestCase):
    """Tests for detector confidence validation."""

    def _make_detector_stub(self):
        """Create a minimal ObjectDetector without loading a real model."""
        from models.detector import ObjectDetector
        with patch.object(ObjectDetector, '_init_model', return_value=None):
            det = ObjectDetector.__new__(ObjectDetector)
            det.config = {'models': {'detector': {}}}
            det.system_info = {}
            det.device = 'cpu'
            det.model = None
            det.confidence = 0.5
        return det

    def test_set_confidence_valid(self):
        """Valid confidence values are accepted."""
        det = self._make_detector_stub()
        det.set_confidence(0.0)
        self.assertAlmostEqual(det.confidence, 0.0)
        det.set_confidence(1.0)
        self.assertAlmostEqual(det.confidence, 1.0)
        det.set_confidence(0.75)
        self.assertAlmostEqual(det.confidence, 0.75)

    def test_set_confidence_out_of_range(self):
        """Out-of-range confidence raises ValueError."""
        det = self._make_detector_stub()
        with self.assertRaises(ValueError):
            det.set_confidence(-0.1)
        with self.assertRaises(ValueError):
            det.set_confidence(1.5)

    def test_set_confidence_wrong_type(self):
        """Non-numeric confidence raises ValueError."""
        det = self._make_detector_stub()
        with self.assertRaises(ValueError):
            det.set_confidence("high")

    def test_detect_without_model(self):
        """Detect returns empty results when model is None."""
        det = self._make_detector_stub()
        import numpy as np
        result = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(len(result['boxes']), 0)
        self.assertEqual(len(result['scores']), 0)


class TestUserManager(unittest.TestCase):
    """Tests for auth/user_manager.py."""

    def test_hash_password_deterministic(self):
        """Same password + salt produces same hash."""
        from auth.user_manager import UserManager
        um = UserManager()
        salt = "abcdef1234567890" * 4  # 64 hex chars
        h1 = um.hash_password("test123", salt)
        h2 = um.hash_password("test123", salt)
        self.assertEqual(h1, h2)

    def test_hash_password_different_salts(self):
        """Different salts produce different hashes."""
        from auth.user_manager import UserManager
        um = UserManager()
        salt1 = "a" * 64
        salt2 = "b" * 64
        h1 = um.hash_password("test123", salt1)
        h2 = um.hash_password("test123", salt2)
        self.assertNotEqual(h1, h2)

    def test_create_and_verify_user(self):
        """Create a user and verify the password."""
        from auth.user_manager import UserManager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            tmppath = f.name
        try:
            um = UserManager({'auth': {'users_file': tmppath}})
            result = um.create_user("testuser", "password123", "user")
            self.assertTrue(result)
            self.assertTrue(um.verify_password("testuser", "password123"))
            self.assertFalse(um.verify_password("testuser", "wrongpassword"))
        finally:
            os.unlink(tmppath)

    def test_delete_user(self):
        """Delete a user and verify it's gone."""
        from auth.user_manager import UserManager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            tmppath = f.name
        try:
            um = UserManager({'auth': {'users_file': tmppath}})
            um.create_user("deletetest", "pass", "user")
            self.assertTrue(um.delete_user("deletetest"))
            self.assertFalse(um.verify_password("deletetest", "pass"))
        finally:
            os.unlink(tmppath)


class TestPreprocessing(unittest.TestCase):
    """Tests for utils/preprocessing.py."""

    def test_is_package_installed(self):
        """Known installed packages are detected."""
        from utils.preprocessing import is_package_installed
        # numpy should be installed in this environment
        self.assertTrue(is_package_installed('numpy'))
        self.assertFalse(is_package_installed('nonexistent_package_xyz_123'))

    def test_check_dependencies_runs(self):
        """check_dependencies runs without error."""
        from utils.preprocessing import check_dependencies
        # Should not raise
        check_dependencies({'os': 'Darwin', 'architecture': 'arm64'})


class TestBaseModels(unittest.TestCase):
    """Tests for models/base.py."""

    def test_base_model_device_detection(self):
        """BaseModel determines device from system_info."""
        from models.base import BaseDetector

        class DummyDetector(BaseDetector):
            def detect(self, image):
                return {}
            def set_confidence(self, c):
                pass

        det = DummyDetector(system_info={'acceleration': 'CPU'})
        self.assertEqual(det.device, 'cpu')

        det2 = DummyDetector(system_info={'acceleration': 'CUDA'})
        self.assertEqual(det2.device, 'cuda')

        det3 = DummyDetector(system_info={'acceleration': 'MPS'})
        self.assertEqual(det3.device, 'mps')


class TestConfigYaml(unittest.TestCase):
    """Tests for config.yaml."""

    def test_config_loads(self):
        """config.yaml can be loaded and has expected sections."""
        import yaml
        config_path = os.path.join(project_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.assertIn('models', config)
        self.assertIn('auth', config)
        self.assertIn('gui', config)
        self.assertIn('processing', config)

    def test_no_absolute_paths_in_config(self):
        """Config should not contain absolute paths."""
        import yaml
        config_path = os.path.join(project_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            content = f.read()
        # Check there are no absolute paths (starting with /)
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Look for path-like values starting with /
            if ': "/' in line or ": '/" in line:
                self.fail(f"Absolute path found in config.yaml: {line}")


class TestPluginManager(unittest.TestCase):
    """Tests for plugins/plugin_manager.py."""

    def test_plugin_manager_init(self):
        """PluginManager initializes without error."""
        from plugins.plugin_manager import PluginManager
        pm = PluginManager({'plugins': {'directory': 'plugins'}})
        self.assertIsInstance(pm.plugins, dict)

    def test_get_nonexistent_plugin(self):
        """Getting a non-existent plugin returns None."""
        from plugins.plugin_manager import PluginManager
        pm = PluginManager({'plugins': {'directory': 'plugins'}})
        self.assertIsNone(pm.get_plugin('nonexistent'))


if __name__ == '__main__':
    unittest.main()
