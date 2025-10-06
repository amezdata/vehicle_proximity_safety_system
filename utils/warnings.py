"""
Warning system module.
Handles warning persistence and display logic.
"""

import cv2
import numpy as np
from avpss.config.settings import (
    WARNING_THRESHOLD, COLORS, FONT, CONFIDENCE_THRESHOLD, ROAD_CLASSES, COLLISION_ALERT_THRESHOLD, CLOSE_DEPTH_THRESHOLD
)


class WarningSystem:
    """
    Manages warning system for close vehicle detection.
    Tracks warning persistence and provides warning display functionality.
    """
    
    def __init__(self):
        """Initialize warning system."""
        # Warning persistence tracking
        self.left_warning_frames = 0
        self.right_warning_frames = 0
        self.warning_threshold = WARNING_THRESHOLD
        
        # Collision alert persistence tracking
        self.collision_alert_frames = 0
        self.collision_alert_threshold = COLLISION_ALERT_THRESHOLD
    
    def update_warning_persistence(self, left_close_vehicles: list, right_close_vehicles: list):
        """
        Update warning persistence counters based on current frame detections.
        
        Args:
            left_close_vehicles: List of close vehicles on left side
            right_close_vehicles: List of close vehicles on right side
        """
        # Update left warning counter
        if left_close_vehicles:
            self.left_warning_frames += 1
        else:
            self.left_warning_frames = 0
        
        # Update right warning counter
        if right_close_vehicles:
            self.right_warning_frames += 1
        else:
            self.right_warning_frames = 0
    
    def update_collision_persistence(self, collision_detected: bool):
        """
        Update collision alert persistence counter.
        
        Args:
            collision_detected: Whether collision was detected in current frame
        """
        if collision_detected:
            self.collision_alert_frames += 1
        else:
            self.collision_alert_frames = 0
    
    def should_show_collision_alert(self) -> bool:
        """
        Check if collision alert should be displayed based on persistence.
        
        Returns:
            bool: True if collision alert should be shown
        """
        return self.collision_alert_frames >= self.collision_alert_threshold
    
    def draw_warnings(self, frame: np.ndarray, frame_width: int, frame_height: int):
        """
        Draw warning messages for close vehicles on left and right sides.
        Only shows warnings if they persist for 3+ consecutive frames.
        
        Args:
            frame: Frame to draw on
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        # Warning colors and settings
        warning_color = COLORS['warning']
        warning_bg_color = COLORS['warning_bg']
        font = getattr(cv2, FONT['family'])
        font_scale = FONT['scale']
        thickness = FONT['thickness']
        
        # Left side warning (only if persistent for 3+ frames)
        if self.left_warning_frames >= self.warning_threshold:
            left_text = "LEFT APPROACH DETECTED"
            left_text_size = cv2.getTextSize(left_text, font, font_scale, thickness)[0]
            
            # Position on left side, middle height
            left_x = 20
            left_y = frame_height // 2
            
            # Draw background rectangle
            cv2.rectangle(frame, (left_x - 10, left_y - left_text_size[1] - 10), 
                        (left_x + left_text_size[0] + 10, left_y + 10), warning_bg_color, -1)
            
            # Draw warning text
            cv2.putText(frame, left_text, (left_x, left_y), font, font_scale, warning_color, thickness)
        
        # Right side warning (only if persistent for 3+ frames)
        if self.right_warning_frames >= self.warning_threshold:
            right_text = "RIGHT APPROACH DETECTED"
            right_text_size = cv2.getTextSize(right_text, font, font_scale, thickness)[0]
            
            # Position on right side, middle height
            right_x = frame_width - right_text_size[0] - 20
            right_y = frame_height // 2
            
            # Draw background rectangle
            cv2.rectangle(frame, (right_x - 10, right_y - right_text_size[1] - 10), 
                        (right_x + right_text_size[0] + 10, right_y + 10), warning_bg_color, -1)
            
            # Draw warning text
            cv2.putText(frame, right_text, (right_x, right_y), font, font_scale, warning_color, thickness)
    
    def draw_collision_alert(self, frame: np.ndarray, frame_width: int, frame_height: int):
        """
        Draw modern collision alert message with animated effects.
        
        Args:
            frame: Frame to draw on
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        # Minimalistic alert design
        alert_text = "COLLISION ALERT"
        
        # Clean colors (BGR format)
        text_color = (255, 255, 255)     # White text
        bg_color = (0, 0, 0)            # Black background
        border_color = (0, 0, 255)       # Red border
        alpha = 0.5                      # Transparency level
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
        
        # Center positioning
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        
        # Create minimal background rectangle
        padding = 20
        rect_top = text_y - text_size[1] - padding
        rect_bottom = text_y + padding
        rect_left = text_x - padding
        rect_right = text_x + text_size[0] + padding
        
        # Draw transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw simple border
        cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), border_color, 2)
        
        # Draw clean text
        cv2.putText(frame, alert_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    def get_close_vehicles_by_side(self, results, depth_map: np.ndarray, frame_width: int, 
                                 class_names: dict) -> tuple:
        """
        Analyze detections and return close vehicles on left and right sides.
        
        Args:
            results: YOLO detection results
            depth_map: Depth map for distance estimation
            frame_width: Width of the frame
            class_names: Dictionary mapping class IDs to names
            
        Returns:
            tuple: (left_close_vehicles, right_close_vehicles)
        """
        left_close_vehicles = []
        right_close_vehicles = []
        
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
                        # Check if close using depth map
                        is_close = False
                        if depth_map is not None:
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
                            is_close = median_depth > CLOSE_DEPTH_THRESHOLD  # Close threshold
                        
                        # Categorize by side
                        if is_close:
                            center_x = (x1 + x2) / 2
                            if center_x < frame_width / 2:
                                left_close_vehicles.append(class_name)
                            else:
                                right_close_vehicles.append(class_name)
        
        return left_close_vehicles, right_close_vehicles
