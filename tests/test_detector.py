"""
Basic test structure for AVPSS detector.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avpss.models.detector import RoadObjectDetector
from avpss.config.settings import ROAD_CLASSES, CONFIDENCE_THRESHOLD


class TestRoadObjectDetector(unittest.TestCase):
    """
    Basic tests for the RoadObjectDetector class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Note: This test requires the YOLO model to be available
        # In a real test environment, you might want to mock this
        try:
            self.detector = RoadObjectDetector()
        except Exception as e:
            self.skipTest(f"Could not initialize detector: {e}")
    
    def test_road_classes_configuration(self):
        """Test that road classes are properly configured."""
        self.assertIsInstance(self.detector.road_classes, dict)
        self.assertIn('car', self.detector.road_classes)
        self.assertIn('truck', self.detector.road_classes)
        self.assertIn('person', self.detector.road_classes)
    
    def test_components_initialization(self):
        """Test that all components are properly initialized."""
        self.assertIsNotNone(self.detector.depth_estimator)
        self.assertIsNotNone(self.detector.roi_manager)
        self.assertIsNotNone(self.detector.visualizer)
        self.assertIsNotNone(self.detector.warning_system)
    
    def test_roi_manager_functionality(self):
        """Test ROI manager basic functionality."""
        # Test ROI coordinates calculation
        roi_coords = self.detector.roi_manager.get_roi_coordinates(800, 600)
        self.assertEqual(len(roi_coords), 4)  # Should return 4 coordinates
        
        # Test ROI intersection
        # Test case: object in ROI
        in_roi = self.detector.roi_manager.is_in_collision_roi(200, 400, 300, 500, 800, 600)
        self.assertIsInstance(in_roi, bool)
        
        # Test case: object outside ROI
        out_roi = self.detector.roi_manager.is_in_collision_roi(50, 100, 150, 200, 800, 600)
        self.assertIsInstance(out_roi, bool)
    
    def test_warning_system_initialization(self):
        """Test warning system initialization."""
        self.assertEqual(self.detector.warning_system.left_warning_frames, 0)
        self.assertEqual(self.detector.warning_system.right_warning_frames, 0)
        self.assertEqual(self.detector.warning_system.warning_threshold, 3)
    
    def test_visualizer_functionality(self):
        """Test visualizer basic functionality."""
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test window name creation
        window_name = self.detector.visualizer.create_window("Test", True)
        self.assertIn("Test", window_name)
        self.assertIn("Depth", window_name)
        
        # Test frame saving (should not raise exception)
        try:
            self.detector.visualizer.save_frame(test_frame, "test_frame.jpg")
            # Clean up
            if os.path.exists("test_frame.jpg"):
                os.remove("test_frame.jpg")
        except Exception as e:
            self.fail(f"Frame saving failed: {e}")


class TestConfiguration(unittest.TestCase):
    """
    Test configuration settings.
    """
    
    def test_road_classes_consistency(self):
        """Test that road classes are consistent with expected values."""
        from avpss.config.settings import ROAD_CLASSES
        
        expected_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
        for class_name in expected_classes:
            self.assertIn(class_name, ROAD_CLASSES)
    
    def test_threshold_values(self):
        """Test that threshold values are reasonable."""
        from avpss.config.settings import CONFIDENCE_THRESHOLD, WARNING_THRESHOLD, COLLISION_THRESHOLD
        
        self.assertGreater(CONFIDENCE_THRESHOLD, 0)
        self.assertLess(CONFIDENCE_THRESHOLD, 1)
        self.assertGreater(WARNING_THRESHOLD, 0)
        self.assertGreater(COLLISION_THRESHOLD, 0)


if __name__ == '__main__':
    unittest.main()
