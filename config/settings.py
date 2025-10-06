"""
Configuration settings for AVPSS system.
Contains all constants and configuration parameters.
"""

# Model Configuration
DEFAULT_MODEL_PATH = 'yolov8n.pt'
MIDAS_MODEL_NAME = 'MiDaS_small'

# Road-specific object classes for detection
ROAD_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'train': 6,
    'truck': 7,
}

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.5
DEPTH_SAMPLE_SIZE = 10
DEPTH_COLLISION_SAMPLE_SIZE = 5

# Warning System Configuration
WARNING_THRESHOLD = 3  # Frames required for warning to appear
COLLISION_THRESHOLD = 600  # Depth threshold for collision alert
COLLISION_ALERT_THRESHOLD = 2  # Frames required for collision alert to appear

# MiDaS Transform Configuration
MIDAS_INPUT_SIZE = 384
MIDAS_MEAN = [0.485, 0.456, 0.406]
MIDAS_STD = [0.229, 0.224, 0.225]

# Distance Categories
CLOSE_DEPTH_THRESHOLD = 1200

# Visualization Colors (BGR format)
COLORS = {
    'truck_bus': (255, 0, 0),      # Blue for trucks and buses
    'motorcycle': (0, 140, 255),      # Orange for motorcycles
    'person': (0, 0, 255),      # Red for persons
    'default': (0, 255, 0),        # Green for other objects
    'warning': (0, 165, 255),     # Orange for warnings
    'warning_bg': (0, 0, 0),      # Black background
    'collision': (0, 0, 255),      # Red for collision alerts
    'collision_bg': (0, 0, 0),     # Black background
    'roi_overlay': (0, 0, 255),    # Light red for ROI
    # Minimalistic alert colors
    'minimal_text': (255, 255, 255),   # White text
    'minimal_bg': (0, 0, 0),           # Black background
    'minimal_border': (0, 0, 255),     # Red border
    'minimal_alpha': 0.5,              # Transparency level (0.0 = fully transparent, 1.0 = opaque)
}

# Font Configuration
FONT = {
    'family': 'FONT_HERSHEY_SIMPLEX',
    'scale': 0.5,
    'thickness': 2,
    'collision_scale': 2.0,
    'collision_thickness': 4,
}

# Grid Configuration
GRID_COLS = 4
GRID_ROWS = 3
ROI_GRID_POSITIONS = [(2, 2), (2, 3), (4, 2), (4, 3)]  # Central squares of bottom row

# Video Configuration
DEFAULT_FOURCC = 'mp4v'
PROGRESS_UPDATE_INTERVAL = 30  # Print progress every N frames

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0

# Display Configuration
SHOW_DETECTIONS = True  # Set to False to hide car detections
SHOW_ROI = False         # Set to False to hide ROI
SHOW_WARNINGS = True     # Keep warnings visible
SHOW_COLLISION_ALERTS = False  # Keep collision alerts visible