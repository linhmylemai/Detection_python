import cv2
import time
from PIL import Image, ImageTk
from object_detection.config import Config
from object_detection.utils.save_log import save_detection Moe

class CameraHandler:
    """Xử lý camera realtime bằng YOLO"""
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.is_paused = False  # Thêm trạng thái pause
        self.detected_objects = []  # danh sách lưu đối tượng phát hiện
        self.paused_frame = None  # Lưu frame cuối cùng khi tạm dừng

    def start_camera(self):
        """Mở camera"""
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.is_paused = False  # Reset trạng thái pause khi mở camera
            self.detected_objects.clear()
            return True
        return False

    def stop_camera(self):
        """Tắt camera và lưu log"""
        if self.cap:
            if self.detected_objects:
                save_detection_log("Camera", "Live Camera", self.detected_objects)
            self.cap.release()
            self.is_running = False
            self.is_paused = False
            self.cap = None
            self.paused_frame = None

    def pause_camera(self):
        """Tạm dừng camera"""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            # Lưu frame hiện tại khi tạm dừng
            self.paused_frame = self.get_frame()

    def resume_camera(self):
        """Tiếp tục camera"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.paused_frame = None

    def get_frame(self):
        """Lấy frame hiện tại từ camera"""
        if self.is_running and self.cap:
            if self.is_paused:
                return self.paused_frame  # Trả về frame tạm dừng nếu đang pause
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def get_processed_frame(self, model):
        """Nhận diện đối tượng trong frame camera"""
        if self.is_paused and self.paused_frame is not None:
            # Sử dụng frame tạm dừng để xử lý
            frame = self.paused_frame
        else:
            frame = self.get_frame()

        if frame is not None:
            start = time.time()

            # ✅ Resize nhỏ để tăng tốc độ xử lý
            small = cv2.resize(frame, (480, 360))

            # ✅ Nhận diện nhanh hơn bằng YOLOv8 (CPU)
            results = model.predict(
                small, verbose=False, conf=0.5, save=False, imgsz=320, device="cpu"
            )

            annotated = results[0].plot()
            fps = 1 / (time.time() - start + 1e-6)

            # ✅ Lưu nhãn và độ chính xác
            if not self.is_paused:  # Chỉ lưu đối tượng khi không tạm dừng
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        conf = float(box.conf[0])
                        self.detected_objects.append((label, conf))

            # ✅ Trả về hình ảnh đã gắn nhãn
            img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize(
                (Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT)
            )
            return ImageTk.PhotoImage(image=img), fps, len(results[0].boxes)
        return None, 0, 0
