"""
Depth estimation module using MiDaS model.
Handles depth map generation and distance estimation.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

from avpss.config.settings import (
    MIDAS_MODEL_NAME, MIDAS_INPUT_SIZE, MIDAS_MEAN, MIDAS_STD,
    DEPTH_SAMPLE_SIZE, CLOSE_DEPTH_THRESHOLD
)


class DepthEstimator:
    """
    Handles depth estimation using MiDaS model.
    Provides depth maps and distance estimation for objects.
    """
    
    def __init__(self):
        """
        Initialize the MiDaS depth estimation model.
        """
        print("Loading MiDaS depth model...")
        self.depth_model = torch.hub.load('intel-isl/MiDaS', MIDAS_MODEL_NAME)
        self.depth_model.eval()
        
        # Setup transforms for MiDaS
        self.transform = transforms.Compose([
            transforms.Resize(MIDAS_INPUT_SIZE),
            transforms.CenterCrop(MIDAS_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MIDAS_MEAN, std=MIDAS_STD)
        ])
        
        print("MiDaS model loaded successfully!")
    
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Get depth map for the input frame using MiDaS.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            depth_map: Depth map as numpy array
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        # Get depth prediction
        with torch.no_grad():
            depth_prediction = self.depth_model(input_tensor)
            depth_map = depth_prediction.squeeze().cpu().numpy()
        
        return depth_map
    
    def estimate_distance(self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
        """
        Estimate relative distance from depth map at bounding box location.
        
        Args:
            depth_map: Depth map from MiDaS
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            distance_category: String indicating relative distance
        """
        # Get center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Sample depth values around the center point
        depth_values = []
        
        for dx in range(-DEPTH_SAMPLE_SIZE, DEPTH_SAMPLE_SIZE, 2):
            for dy in range(-DEPTH_SAMPLE_SIZE, DEPTH_SAMPLE_SIZE, 2):
                sample_x = max(0, min(depth_map.shape[1]-1, center_x + dx))
                sample_y = max(0, min(depth_map.shape[0]-1, center_y + dy))
                depth_values.append(depth_map[sample_y, sample_x])
        
        # Get median depth value (more robust than mean)
        median_depth = np.median(depth_values)
        
        # Convert to relative distance categories
        # Higher depth values = closer objects in MiDaS
        if median_depth > CLOSE_DEPTH_THRESHOLD:
            return f"Close ({median_depth:.2f})"
        else:
            return f"Normal ({median_depth:.2f})"
    
    def get_depth_at_location(self, depth_map: np.ndarray, center_x: int, center_y: int, sample_size: int = 5) -> float:
        """
        Get median depth value at a specific location with sampling.
        
        Args:
            depth_map: Depth map from MiDaS
            center_x, center_y: Center coordinates
            sample_size: Size of sampling area around center
            
        Returns:
            median_depth: Median depth value at the location
        """
        depth_values = []
        
        for dx in range(-sample_size, sample_size, 2):
            for dy in range(-sample_size, sample_size, 2):
                sample_x = max(0, min(depth_map.shape[1]-1, center_x + dx))
                sample_y = max(0, min(depth_map.shape[0]-1, center_y + dy))
                depth_values.append(depth_map[sample_y, sample_x])
        
        return np.median(depth_values)
