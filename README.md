# AVPSS - Advanced Vehicle Proximity Safety System

## Author
R.H. Amezqueta - [LinkedIn](https://www.linkedin.com/in/rhamezqueta)

## License
All rights reserved. This project is for portfolio/demonstration purposes.
Contact the author for permission to use or modify this code.

A modular system for road object detection with depth estimation using YOLO and MiDaS.

## Video Presentation
https://youtu.be/TFD78DyHoFk

## Project Structure

```
avpss/
├── __init__.py
├── main.py                    # Main entry point
├── example_usage.py           # Usage examples and tutorials
├── models/
│   ├── __init__.py
│   ├── detector.py           # Main RoadObjectDetector class
│   └── depth_estimator.py    # MiDaS depth estimation
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Drawing functions
│   ├── roi_manager.py        # ROI and collision detection
│   └── warnings.py           # Warning system
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration constants
└── tests/
    ├── __init__.py
    └── test_detector.py      # Basic tests
```

## Key Components

### 1. **RoadObjectDetector** (`models/detector.py`)
- Main coordinator class
- Handles video/camera input
- Coordinates all other components
- Configurable display options

### 2. **DepthEstimator** (`models/depth_estimator.py`)
- MiDaS depth model integration
- Distance estimation for objects
- Depth map generation

### 3. **ROIManager** (`utils/roi_manager.py`)
- Region of Interest calculations
- Collision detection logic
- ROI intersection testing

### 4. **Visualizer** (`utils/visualization.py`)
- Drawing and annotation functions
- Frame display management
- Visualization utilities

### 5. **WarningSystem** (`utils/warnings.py`)
- Warning persistence tracking
- Alert display logic
- Close vehicle detection
- Collision alert persistence

### 6. **Configuration** (`config/settings.py`)
- All constants and parameters
- Color schemes and thresholds
- Model configurations
- Display control flags

## Features

- ✅ **Real-time object detection** using YOLO
- ✅ **Depth estimation** using MiDaS
- ✅ **Collision detection** with ROI analysis
- ✅ **Warning system** with persistence
- ✅ **Configurable display** options
- ✅ **Modular architecture** for easy maintenance
- ✅ **Transparent alerts** with modern design

## Usage

### Basic Usage
```bash
# Run with camera (default)
python -m avpss.main

# Run with video file
python -m avpss.main --mode video --path sample_videos/accidents.mp4

# Run without depth estimation (faster)
python -m avpss.main --no-depth

# Save output video
python -m avpss.main --mode video --path input.mp4 --output output.mp4

# Show current display configuration
python -m avpss.main --show-config
```

### Command Line Options
- `--mode`: Choose 'video' or 'camera' mode
- `--path`: Path to video file (required for video mode)
- `--output`: Output path for processed video
- `--model`: Path to YOLO model weights
- `--camera`: Camera index for camera mode
- `--no-display`: Disable video display
- `--no-depth`: Disable depth estimation
- `--show-config`: Show current display configuration

### Display Configuration

The system supports configurable display options in `config/settings.py`:

```python
# Display Configuration
SHOW_DETECTIONS = False  # Hide car bounding boxes
SHOW_ROI = False         # Hide collision zone overlay
SHOW_WARNINGS = True     # Show approach warnings
SHOW_COLLISION_ALERTS = True  # Show collision alerts
```

**Current default**: Warnings and collision alerts only (clean display)

### Example Usage

```bash
# Run interactive examples
python avpss/example_usage.py

# Options available:
# 1. Camera detection
# 2. Video processing (warnings only)
# 3. Component usage
# 4. Display configuration help
# 5. Custom batch processing
```

## Dependencies

- ultralytics (YOLO)
- torch
- torchvision
- opencv-python
- numpy
- PIL (Pillow)

## Testing

```bash
# Run basic tests
python -m avpss.tests.test_detector
```

## Limitations and Disclaimers

⚠️ **IMPORTANT**: This is a demonstration/portfolio project and is **NOT suitable for actual vehicle deployment**.

### Key Limitations:

- **No Real-time Safety Guarantees**: This system is for demonstration purposes only
- **Limited Testing**: Only basic tests are included; comprehensive safety testing is not performed
- **No Hardware Integration**: Designed for desktop/laptop use with webcams, not vehicle integration
- **No Safety Standards Compliance**: Does not meet automotive safety standards or regulations
- **Performance Limitations**: Not optimized for real-time vehicle safety requirements
- **No Redundancy**: Lacks the redundancy and fail-safe mechanisms required for vehicle safety systems
- **Limited Object Detection**: Only detects basic road objects; may miss critical safety scenarios
- **No Environmental Adaptation**: Not tested across various weather, lighting, and road conditions

### Intended Use:
- **Portfolio demonstration** of computer vision and AI skills
- **Educational purposes** for learning object detection and depth estimation
- **Research and development** for understanding proximity detection concepts
- **Academic projects** and technical demonstrations

### NOT Intended For:
- ❌ Actual vehicle safety systems
- ❌ Production automotive applications
- ❌ Real-time safety-critical operations
- ❌ Commercial vehicle deployment
- ❌ Any safety-critical applications

**Use at your own risk. The author assumes no responsibility for any consequences of using this software.**



