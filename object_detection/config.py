CONFIDENCE_THRESHOLD = 0.5

class Config:
    # Window settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 700
    MENU_WIDTH = 300

    # YOLO model
    MODEL_PATH = "yolov8n.pt"

    # Data directories
    DATA_DIR = "data"
    INPUTS = "inputs"
    OUTPUTS = "outputs"

    # Image display
    IMAGE_W = 900
    IMAGE_H = 700

    # Frame rate and timing
    FRAME_DELAY = 20  # ms
    MIN_FPS = 1e-6
    # Supported file types
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    VIDEO_EXTS = (".mp4", ".avi", ".mov")
    
