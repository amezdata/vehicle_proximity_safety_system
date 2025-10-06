"""
Main detector class for AVPSS system.
Coordinates all components for road object detection with depth estimation.
"""

import cv2
import os
import sys
from pathlib import Path
import numpy as np

# Try to import YOLO with error handling
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

from avpss.models.depth_estimator import DepthEstimator
from avpss.utils.roi_manager import ROIManager
from avpss.utils.visualization import Visualizer
from avpss.utils.warnings import WarningSystem
from avpss.config.settings import (
    DEFAULT_MODEL_PATH, ROAD_CLASSES, PROGRESS_UPDATE_INTERVAL, 
    DEFAULT_FOURCC, DEFAULT_CAMERA_INDEX, SHOW_DETECTIONS, SHOW_ROI, 
    SHOW_WARNINGS, SHOW_COLLISION_ALERTS
)


class RoadObjectDetector:
    """
    Main detector class for road object detection with depth estimation.
    Coordinates all components: YOLO detection, depth estimation, ROI management,
    visualization, and warning systems.
    """
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Initialize the detector with all required components.
        
        Args:
            model_path: Path to YOLO model weights
        """
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Initialize components
        self.depth_estimator = DepthEstimator()
        self.roi_manager = ROIManager()
        self.visualizer = Visualizer()
        self.warning_system = WarningSystem()
        
        # Road-specific object classes
        self.road_classes = ROAD_CLASSES
        
        print("AVPSS Road Object Detector initialized successfully!")
    
    def detect_from_video(self, video_path: str, output_path: str = None, 
                         display: bool = True, use_depth: bool = True):
        """
        Detect objects in a video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            display: Whether to display video during processing
            use_depth: Whether to use depth estimation
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*DEFAULT_FOURCC)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame = self._process_frame(frame, use_depth)
            
            # Save frame if output is specified
            if out:
                out.write(annotated_frame)
            
            # Display frame
            if display:
                window_title = self.visualizer.create_window('AVPSS - Road Object Detection', use_depth)
                cv2.imshow(window_title, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % PROGRESS_UPDATE_INTERVAL == 0:
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
                processing_fps = frame_count / elapsed_time
                print(f"Processed {frame_count} frames... Processing FPS: {processing_fps:.2f}")
        
        # Calculate final processing FPS
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency()
        final_fps = frame_count / total_time
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing completed. Processed {frame_count} frames.")
        print(f"Average processing FPS: {final_fps:.2f}")
        if output_path:
            print(f"Output saved to: {output_path}")
    
    def detect_from_camera(self, camera_index: int = DEFAULT_CAMERA_INDEX, 
                           display: bool = True, use_depth: bool = True):
        """
        Detect objects from camera feed in real-time.
        
        Args:
            camera_index: Camera index (0 for default camera)
            display: Whether to display video feed
            use_depth: Whether to use depth estimation
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        print(f"Starting real-time detection from camera {camera_index}")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Process frame
            annotated_frame = self._process_frame(frame, use_depth)
            
            # Add frame counter
            self.visualizer.draw_frame_counter(annotated_frame, frame_count)
            
            # Display frame
            if display:
                window_title = self.visualizer.create_window('AVPSS - Real-time Road Object Detection', use_depth)
                cv2.imshow(window_title, annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"detection_frame_{frame_count}.jpg"
                    if self.visualizer.save_frame(annotated_frame, filename):
                        print(f"Frame saved as: {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection stopped.")
    
    def _process_frame(self, frame: np.ndarray, use_depth: bool = True) -> np.ndarray:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Input frame
            use_depth: Whether to use depth estimation
            
        Returns:
            Annotated frame with detections and warnings
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Get depth map for distance estimation if enabled
        depth_map = None
        if use_depth:
            depth_map = self.depth_estimator.get_depth_map(frame)
        
        # Get close vehicles for warning system
        left_close_vehicles, right_close_vehicles = self.warning_system.get_close_vehicles_by_side(
            results, depth_map, frame.shape[1], self.class_names
        )
        
        # Update warning persistence
        self.warning_system.update_warning_persistence(left_close_vehicles, right_close_vehicles)
        
        # Check for collision alert in ROI
        collision_detected = False
        if depth_map is not None:
            collision_detected = self.roi_manager.check_collision_alert(
                frame, results, depth_map, frame.shape[1], frame.shape[0], self.class_names
            )
        
        # Update collision alert persistence
        self.warning_system.update_collision_persistence(collision_detected)
        
        # Start with original frame
        annotated_frame = frame.copy()
        
        # Draw detections with depth information (if enabled)
        if SHOW_DETECTIONS:
            annotated_frame = self.visualizer.draw_detections(frame, results, depth_map, self.class_names)
        
        # Draw ROI visualization (if enabled)
        if SHOW_ROI:
            self.visualizer.draw_roi(annotated_frame, frame.shape[1], frame.shape[0])
        
        # Draw warnings for close vehicles (if enabled)
        if SHOW_WARNINGS:
            self.warning_system.draw_warnings(annotated_frame, frame.shape[1], frame.shape[0])
        
        # Draw collision alert only if it persists for 2+ frames (if enabled)
        if SHOW_COLLISION_ALERTS and self.warning_system.should_show_collision_alert():
            self.warning_system.draw_collision_alert(annotated_frame, frame.shape[1], frame.shape[0])
        
        return annotated_frame