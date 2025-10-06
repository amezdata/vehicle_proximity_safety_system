"""
Main entry point for AVPSS system.
Handles command line arguments and coordinates the detection system.
"""

import argparse
import sys
from avpss.models.detector import RoadObjectDetector
from avpss.config.settings import (
    DEFAULT_MODEL_PATH, DEFAULT_CAMERA_INDEX, 
    SHOW_DETECTIONS, SHOW_ROI, SHOW_WARNINGS, SHOW_COLLISION_ALERTS
)


def main():
    """
    Main function to run the AVPSS object detector with depth estimation.
    """
    parser = argparse.ArgumentParser(description='AVPSS Road Object Detection with MiDaS Depth Estimation')
    parser.add_argument('--mode', choices=['video', 'camera'], default='camera',
                       help='Detection mode: video file or camera')
    parser.add_argument('--path', type=str, default=None,
                       help='Path to video file (required for video mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for processed video')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to YOLO model weights')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX,
                       help='Camera index for camera mode')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--no-depth', action='store_true',
                       help='Disable depth estimation (faster processing)')
    parser.add_argument('--show-config', action='store_true',
                       help='Show current display configuration and exit')
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        print("AVPSS Display Configuration")
        print("==========================")
        print(f"Show detections: {SHOW_DETECTIONS}")
        print(f"Show ROI: {SHOW_ROI}")
        print(f"Show warnings: {SHOW_WARNINGS}")
        print(f"Show collision alerts: {SHOW_COLLISION_ALERTS}")
        print()
        print("To modify these settings, edit avpss/config/settings.py")
        return
    
    # Initialize detector
    print("Initializing AVPSS Road Object Detector with MiDaS Depth Estimation...")
    print(f"Display mode: {'Warnings only' if not SHOW_DETECTIONS and not SHOW_ROI else 'Full detection'}")
    
    try:
        detector = RoadObjectDetector(args.model)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    try:
        if args.mode == 'video':
            if not args.path:
                print("Error: --path is required for video mode")
                return
            
            print(f"Processing video: {args.path}")
            detector.detect_from_video(
                video_path=args.path,
                output_path=args.output,
                display=not args.no_display,
                use_depth=not args.no_depth
            )
        
        elif args.mode == 'camera':
            print("Starting camera detection...")
            detector.detect_from_camera(
                camera_index=args.camera,
                display=not args.no_display,
                use_depth=not args.no_depth
            )
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    except Exception as e:
        print(f"Error during detection: {e}")
        return


if __name__ == "__main__":
    main()