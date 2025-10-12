import os
import shutil
from ultralytics import YOLO
from object_detection.utils.image_utils import detect_image, save_image
from object_detection.config import Config


# --- Chuẩn bị đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, Config.DATA_DIR)
INPUT_DIR = os.path.join(DATA_DIR, Config.INPUTS)
OUTPUT_DIR = os.path.join(DATA_DIR, Config.OUTPUTS)
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", Config.MODEL_PATH)

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

def detect_from_folder():
    """Nhận diện tất cả ảnh trong thư mục inputs."""
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(Config.IMAGE_EXTS):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            annotated = detect_image(model, input_path)
            if annotated is not None:
                save_image(annotated, output_path)
                print(f"{filename} -> Done")
            else:
                print(f"Bỏ qua: {filename}")
    print("✅ Hoàn tất nhận diện thư mục inputs!")

if __name__ == "__main__":
    detect_from_folder()
