"""
ROI (Region of Interest) management module.
Handles collision detection and ROI calculations.
"""

import numpy as np
from avpss.config.settings import (
    GRID_COLS, GRID_ROWS, ROI_GRID_POSITIONS, COLLISION_THRESHOLD,
    CONFIDENCE_THRESHOLD, ROAD_CLASSES, DEPTH_COLLISION_SAMPLE_SIZE
)


class ROIManager:
    """
    Manages Region of Interest calculations and collision detection.
    """
    
    def __init__(self):
        """Initialize ROI manager."""
        self.collision_roi_frames = 0
    
    def is_in_collision_roi(self, x1: int, y1: int, x2: int, y2: int, 
                           frame_width: int, frame_height: int) -> bool:
        """
        Check if a vehicle bounding box intersects with the collision ROI.
        ROI is the two central squares of the bottom row: (2,2), (2,3), (4,2), (4,3)
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            bool: True if vehicle intersects with collision ROI
        """
        # Calculate grid dimensions
        grid_width = frame_width // GRID_COLS
        grid_height = frame_height // GRID_ROWS
        
        # ROI is the two central squares of the bottom row
        # Grid positions: (2,2), (2,3), (4,2), (4,3)
        # This means columns 1-2 (0-indexed) of the bottom row
        roi_top = frame_height - grid_height
        roi_bottom = frame_height
        roi_left = grid_width  # Start from column 1 (0-indexed)
        roi_right = 3 * grid_width  # End at column 3 (0-indexed)
        
        # Check if bounding box intersects with ROI
        return not (x2 < roi_left or x1 > roi_right or y2 < roi_top or y1 > roi_bottom)
    
    def check_collision_alert(self, frame: np.ndarray, results, depth_map: np.ndarray, 
                            frame_width: int, frame_height: int, class_names: dict) -> bool:
        """
        Check for collision alert conditions in the ROI.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            depth_map: Depth map for distance estimation
            frame_width: Width of the frame
            frame_height: Height of the frame
            class_names: Dictionary mapping class IDs to names
            
        Returns:
            bool: True if collision alert should be shown
        """
        collision_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id]
                    
                    # Only check road-relevant objects with good confidence
                    if confidence > CONFIDENCE_THRESHOLD and class_id in ROAD_CLASSES.values():
                        # Check if vehicle is in collision ROI
                        if self.is_in_collision_roi(x1, y1, x2, y2, frame_width, frame_height):
                            if depth_map is not None:
                                # Get depth value at vehicle location
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                # Sample depth values around center
                                depth_values = []
                                for dx in range(-DEPTH_COLLISION_SAMPLE_SIZE, DEPTH_COLLISION_SAMPLE_SIZE, 2):
                                    for dy in range(-DEPTH_COLLISION_SAMPLE_SIZE, DEPTH_COLLISION_SAMPLE_SIZE, 2):
                                        sample_x = max(0, min(depth_map.shape[1]-1, center_x + dx))
                                        sample_y = max(0, min(depth_map.shape[0]-1, center_y + dy))
                                        depth_values.append(depth_map[sample_y, sample_x])
                                
                                median_depth = np.median(depth_values)
                                
                                # Check collision condition
                                if median_depth > COLLISION_THRESHOLD:
                                    collision_detected = True
                                    break
        
        return collision_detected
    
    def get_roi_coordinates(self, frame_width: int, frame_height: int) -> tuple:
        """
        Get ROI coordinates for visualization.
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            tuple: (roi_left, roi_top, roi_right, roi_bottom)
        """
        # Calculate grid dimensions
        grid_width = frame_width // GRID_COLS
        grid_height = frame_height // GRID_ROWS
        
        # ROI is the two central squares of the bottom row
        roi_top = frame_height - grid_height
        roi_bottom = frame_height
        roi_left = grid_width  # Start from column 1 (0-indexed)
        roi_right = 3 * grid_width  # End at column 3 (0-indexed)
        
        return roi_left, roi_top, roi_right, roi_bottom
