"""
Example usage of the modular AVPSS system.
Demonstrates how to use the system programmatically with display configuration options.
"""

import sys
import os
from pathlib import Path

# Add the avpss module to path
sys.path.insert(0, str(Path(__file__).parent))

from avpss.models.detector import RoadObjectDetector
from avpss.config.settings import (
    DEFAULT_MODEL_PATH, SHOW_DETECTIONS, SHOW_ROI, SHOW_WARNINGS, SHOW_COLLISION_ALERTS
)


def example_camera_detection():
    """Example of using the detector with camera."""
    print("Initializing AVPSS detector...")
    
    try:
        # Initialize detector
        detector = RoadObjectDetector(DEFAULT_MODEL_PATH)
        
        # Start camera detection
        print("Starting camera detection...")
        print(f"Current display settings:")
        print(f"  - Show detections: {SHOW_DETECTIONS}")
        print(f"  - Show ROI: {SHOW_ROI}")
        print(f"  - Show warnings: {SHOW_WARNINGS}")
        print(f"  - Show collision alerts: {SHOW_COLLISION_ALERTS}")
        
        detector.detect_from_camera(
            camera_index=0,
            display=True,
            use_depth=True
        )
        
    except Exception as e:
        print(f"Error: {e}")


def example_video_processing():
    """Example of processing a video file."""
    print("Initializing AVPSS detector...")
    
    try:
        # Initialize detector
        detector = RoadObjectDetector(DEFAULT_MODEL_PATH)
        
        # Process video
        video_path = "sample_videos/accidents.mp4"
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            print(f"Display mode: Warnings only (detections and ROI hidden)")
            
            detector.detect_from_video(
                video_path=video_path,
                output_path="output_warnings_only.mp4",
                display=True,
                use_depth=True
            )
        else:
            print(f"Video file not found: {video_path}")
            print("Please ensure sample_videos/accidents.mp4 exists")
            
    except Exception as e:
        print(f"Error: {e}")


def example_component_usage():
    """Example of using individual components."""
    from avpss.models.depth_estimator import DepthEstimator
    from avpss.utils.roi_manager import ROIManager
    from avpss.utils.warnings import WarningSystem
    from avpss.utils.visualization import Visualizer
    
    print("Initializing individual components...")
    
    try:
        # Initialize components
        depth_estimator = DepthEstimator()
        roi_manager = ROIManager()
        warning_system = WarningSystem()
        visualizer = Visualizer()
        
        print("All components initialized successfully!")
        
        # Example of using ROI manager
        frame_width, frame_height = 800, 600
        roi_coords = roi_manager.get_roi_coordinates(frame_width, frame_height)
        print(f"ROI coordinates: {roi_coords}")
        
        # Example of checking ROI intersection
        in_roi = roi_manager.is_in_collision_roi(200, 400, 300, 500, frame_width, frame_height)
        print(f"Object in ROI: {in_roi}")
        
        # Example of warning system persistence
        print(f"Warning threshold: {warning_system.warning_threshold} frames")
        print(f"Collision alert threshold: {warning_system.collision_alert_threshold} frames")
        
    except Exception as e:
        print(f"Error: {e}")


def example_display_configuration():
    """Example of how to modify display configuration."""
    print("Display Configuration Examples")
    print("=============================")
    print()
    print("To modify what's displayed, edit avpss/config/settings.py:")
    print()
    print("# Show everything")
    print("SHOW_DETECTIONS = True")
    print("SHOW_ROI = True")
    print("SHOW_WARNINGS = True")
    print("SHOW_COLLISION_ALERTS = True")
    print()
    print("# Show only warnings (current setting)")
    print("SHOW_DETECTIONS = False  # Hide car bounding boxes")
    print("SHOW_ROI = False         # Hide collision zone")
    print("SHOW_WARNINGS = True     # Show approach warnings")
    print("SHOW_COLLISION_ALERTS = True  # Show collision alerts")
    print()
    print("# Show only collision alerts")
    print("SHOW_DETECTIONS = False")
    print("SHOW_ROI = False")
    print("SHOW_WARNINGS = False")
    print("SHOW_COLLISION_ALERTS = True")
    print()
    print("Current settings:")
    print(f"  - Show detections: {SHOW_DETECTIONS}")
    print(f"  - Show ROI: {SHOW_ROI}")
    print(f"  - Show warnings: {SHOW_WARNINGS}")
    print(f"  - Show collision alerts: {SHOW_COLLISION_ALERTS}")


def example_custom_processing():
    """Example of custom video processing with different settings."""
    print("Custom Video Processing Example")
    print("===============================")
    
    try:
        # Initialize detector
        detector = RoadObjectDetector(DEFAULT_MODEL_PATH)
        
        # Process multiple videos with different settings
        videos = [
            ("sample_videos/accidents.mp4", "accidents_processed.mp4"),
            ("sample_videos/dashcam.mp4", "dashcam_processed.mp4")
        ]
        
        for input_video, output_video in videos:
            if os.path.exists(input_video):
                print(f"Processing {input_video} -> {output_video}")
                detector.detect_from_video(
                    video_path=input_video,
                    output_path=output_video,
                    display=False,  # No display for batch processing
                    use_depth=True
                )
            else:
                print(f"Video not found: {input_video}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("AVPSS Example Usage")
    print("==================")
    print("1. Camera detection")
    print("2. Video processing (warnings only)")
    print("3. Component usage")
    print("4. Display configuration help")
    print("5. Custom batch processing")
    
    choice = input("Select example (1-5): ")
    
    if choice == "1":
        example_camera_detection()
    elif choice == "2":
        example_video_processing()
    elif choice == "3":
        example_component_usage()
    elif choice == "4":
        example_display_configuration()
    elif choice == "5":
        example_custom_processing()
    else:
        print("Invalid choice")