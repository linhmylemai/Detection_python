import os

class Config:
    # Window settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 700
    MENU_WIDTH = 300

    # YOLO model
    MODEL_PATH = "yolov8n.pt"

    # Base directory: luôn trỏ đến "object_detection"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Data directories (đảm bảo nằm trong object_detection/data/)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUTS = os.path.join(DATA_DIR, "inputs")
    OUTPUTS = os.path.join(DATA_DIR, "outputs")

    # Asset/model directories (đảm bảo nằm trong object_detection/assets)
    ASSETS_DIR = os.path.join(BASE_DIR,"assets")
    MODELS_DIR = os.path.join(ASSETS_DIR, "models")

    # Image display
    IMAGE_W = 1000
    IMAGE_H = 700

    # Frame rate and timing
    FRAME_DELAY = 20  # ms
    MIN_FPS = 1e-6

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.5

    # Supported file types
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")
    VIDEO_EXTS = (".mp4", ".avi", ".mov")
