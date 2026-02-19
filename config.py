# --- File for configuration constants for the Hand Tracking Disney Channel project ---

# Camera settings
CAMERA_INDEX = 0          # 0 main webcam
FRAME_WIDTH  = 1280       # Wide - frame width in pixels
FRAME_HEIGHT = 720        # High - frame height in pixels
FPS_TARGET   = 30         # FPS 

# Number of hand to track 
MAX_HANDS = 1

# Confidence settings for MediaPipe - IA parameters
DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.7

# Initian colors (BGR format for OpenCV)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0)
COLOR_BLUE    = (255, 100, 0)    
COLOR_YELLOW  = (0,   220, 255)  
COLOR_GREEN   = (0,   255, 0)    
COLOR_RED     = (0,   0,   255)  


WRIST           = 0
THUMB_TIP       = 4
INDEX_TIP       = 8   # ← la "varita"
MIDDLE_TIP      = 12
RING_TIP        = 16
PINKY_TIP       = 20

# Window settings for OpenCV display
WINDOW_NAME = "Disney Channel — Hand Tracking"