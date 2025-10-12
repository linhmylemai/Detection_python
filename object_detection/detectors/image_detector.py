import os
import shutil
from ultralytics import YOLO
from utils.image_utils import detect_image, save_image
from config import Config


os.makedirs(Config.INPUTS, exist_ok=True)
os.makedirs(Config.OUTPUTS, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(Config.MODEL_PATH)
print(f"✅ Model loaded from {Config.MODEL_PATH}")

# --- Nhận diện một ảnh đơn (dùng cho GUI) ---
def detect_single_image(file_path):
    """Nhận diện ảnh đơn lẻ và lưu kết quả vào thư mục outputs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {file_path}")

    annotated = detect_image(model, file_path)
    if annotated is None:
        return None

    # --- Lưu kết quả ---
    output_path = os.path.join(Config.OUTPUTS, os.path.basename(file_path))
    save_image(annotated, output_path)

    # --- Copy ảnh gốc vào thư mục inputs (nếu chưa có) ---
    dest_input_path = os.path.join(Config.INPUTS, os.path.basename(file_path))
    if not os.path.exists(dest_input_path):
        try:
            shutil.copy(file_path, dest_input_path)
        except Exception as e:
            print(f"Lỗi khi copy ảnh: {e}")

    return output_path  # trả về đường dẫn ảnh kết quả
