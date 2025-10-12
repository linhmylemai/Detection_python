import os
import shutil
from ultralytics import YOLO
from utils.image_utils import detect_image, save_image
from config import Config


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

# --- Nhận diện một ảnh đơn (dùng cho GUI) ---
def detect_single_image(file_path):
    """Nhận diện ảnh đơn lẻ và lưu kết quả vào thư mục outputs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {file_path}")

    annotated = detect_image(model, file_path)
    if annotated is None:
        return None

    # --- Lưu kết quả ---
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    save_image(annotated, output_path)

    # --- Copy ảnh gốc vào thư mục inputs (nếu chưa có) ---
    dest_input_path = os.path.join(INPUT_DIR, os.path.basename(file_path))
    if not os.path.exists(dest_input_path):
        try:
            shutil.copy(file_path, dest_input_path)
        except Exception as e:
            print(f"Lỗi khi copy ảnh: {e}")

    return output_path  # trả về đường dẫn ảnh kết quả
