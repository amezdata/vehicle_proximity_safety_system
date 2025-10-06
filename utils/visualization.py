"""
Visualization module.
Handles all drawing and display functionality.
"""

import cv2
import numpy as np
from avpss.config.settings import (
    COLORS, FONT, CONFIDENCE_THRESHOLD, ROAD_CLASSES, GRID_COLS, GRID_ROWS, CLOSE_DEPTH_THRESHOLD
)


class Visualizer:
    """
    Handles all visualization and drawing operations.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        pass
    
    def draw_detections(self, frame: np.ndarray, results, depth_map: np.ndarray = None, 
                       class_names: dict = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame with depth information.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            depth_map: Optional depth map for distance estimation
            class_names: Dictionary mapping class IDs to names
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id] if class_names else f"class_{class_id}"
                    
                    # Only draw if confidence is above threshold and it's a road-relevant object
                    if confidence > CONFIDENCE_THRESHOLD and class_id in ROAD_CLASSES.values():
                        # Estimate distance if depth map is available
                        distance_info = ""
                        if depth_map is not None:
                            # Simple distance estimation based on depth map
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Sample depth values around center
                            depth_values = []
                            for dx in range(-5, 5, 2):
                                for dy in range(-5, 5, 2):
                                    sample_x = max(0, min(depth_map.shape[1]-1, center_x + dx))
                                    sample_y = max(0, min(depth_map.shape[0]-1, center_y + dy))
                                    depth_values.append(depth_map[sample_y, sample_x])
                            
                            median_depth = np.median(depth_values)
                            if median_depth > CLOSE_DEPTH_THRESHOLD:
                                distance_info = f" | Close ({median_depth:.2f})"
                            else:
                                distance_info = f" | Normal ({median_depth:.2f})"
                        
                        # Determine color based on object type
                        if class_name in ['truck', 'bus']:
                            color = COLORS['truck_bus']
                        elif class_name in [ 'motorcycle']:
                            color = COLORS['motorcycle']
                        elif class_name in ['person']:
                            color = COLORS['person']
                        else:
                            color = COLORS['default']
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label with confidence and distance
                        label = f"{class_name}: {confidence:.2f}{distance_info}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def draw_roi(self, frame: np.ndarray, frame_width: int, frame_height: int):
        """
        Draw the collision ROI visualization (two central squares of bottom row).
        
        Args:
            frame: Frame to draw on
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        # Calculate grid dimensions
        grid_width = frame_width // GRID_COLS
        grid_height = frame_height // GRID_ROWS
        
        # ROI is the two central squares of the bottom row
        roi_top = frame_height - grid_height
        roi_bottom = frame_height
        roi_left = grid_width  # Start from column 1 (0-indexed)
        roi_right = 3 * grid_width  # End at column 3 (0-indexed)
        
        # Draw ROI rectangle with semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_left, roi_top), (roi_right, roi_bottom), COLORS['roi_overlay'], -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)  # Semi-transparent
    
    def draw_frame_counter(self, frame: np.ndarray, frame_count: int):
        """
        Draw frame counter on the frame.
        
        Args:
            frame: Frame to draw on
            frame_count: Current frame number
        """
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def draw_fps(self, frame: np.ndarray, fps: float):
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS value
        """
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def create_window(self, window_name: str, use_depth: bool = True) -> str:
        """
        Create and return appropriate window name.
        
        Args:
            window_name: Base window name
            use_depth: Whether depth estimation is enabled
            
        Returns:
            str: Full window name
        """
        if use_depth:
            return f"{window_name} with Depth"
        else:
            return f"{window_name}"
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save current frame to file.
        
        Args:
            frame: Frame to save
            filename: Output filename
            
        Returns:
            bool: True if successful
        """
        try:
            cv2.imwrite(filename, frame)
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
