import cv2
import time
import os
from ultralytics import YOLO
from object_detection.utils.save_log import save_detection_log
from object_detection.config import Config

class VideoDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Khởi tạo VideoDetector với mô hình YOLO.

        Args:
            model_path (str): Đường dẫn đến file mô hình YOLO.
        """
        print("🚀 Đang load model YOLO (CPU)...")
        self.model = YOLO(model_path)
        self.cap = None
        self.running_mode = None
        self.paused = False
        self.last_frame = None  # Thêm để lưu frame cuối cùng khi tạm dừng

    def detect_video(self, video_path: str) -> bool:
        """
        Mở và bắt đầu xử lý video từ đường dẫn file.
        Args:
            video_path (str): Đường dẫn đến file video.
        Returns:
            bool: True nếu mở video thành công, False nếu thất bại.
        """
        if not os.path.exists(video_path):
            print(f"❌ Không tìm thấy video: {video_path}")
            return False

        self.stop()
        self.current_video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.running_mode = "video"
        if not self.cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}. Kiểm tra định dạng hoặc codec.")
            return False
        try:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
            print(f"📹 Video codec: {codec}")
        except Exception as e:
            print(f"⚠️ Không thể kiểm tra codec: {str(e)}")
        return True

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        print(f"⏯ Video {'tạm dừng' if self.paused else 'tiếp tục'}.")
        return self.paused
    
    def process_stream(self) -> tuple:
        if self.cap is None or self.running_mode not in ("video", "camera"):
            print("⚠️ Không có stream để xử lý.")
            return None, None, None
        
        if self.paused:
            if self.last_frame is not None:
                return self.last_frame, None, 0.0
            return None, None, None

        ret, frame = self.cap.read()
        if not ret:
            print("📁 Video đã kết thúc.")
            return None, None, None

        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = self.model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)

        self.last_frame = annotated  # Lưu frame cuối cùng
        detected_objects = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                detected_objects.append((label, conf))
        if detected_objects:
            source = self.current_video_path if self.running_mode == "video" else "Live Stream"
            save_detection_log(self.running_mode.capitalize(), source, detected_objects)

        return annotated, results, fps
    
    def stop(self):
        """
        Dừng video hoặc camera và giải phóng tài nguyên.
        """
        self.running_mode = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        print("⏹ Stream đã dừng.")