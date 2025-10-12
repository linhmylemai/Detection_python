import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time, shutil
from detectors.image_detector import detect_single_image  
from config import Config

# ===== Thiết lập thư mục =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, Config.DATA_DIR)
INPUT_DIR = os.path.join(DATA_DIR, Config.INPUTS)
OUTPUT_DIR = os.path.join(DATA_DIR, Config.OUTPUTS)
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", Config.MODEL_PATH)

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load YOLO model =====
model = YOLO(MODEL_PATH)

# ===== Giao diện chính =====
root = tb.Window(themename="cosmo")
root.title("Object Detection App")
root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

# ===== Ảnh nền =====
bg_path = os.path.join(BASE_DIR, "assets", "Background.png")
bg_image = Image.open(bg_path).resize((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
bg_photo = ImageTk.PhotoImage(bg_image)

# Tạo label nền và đặt ở dưới cùng
background_label = tb.Label(root, image=bg_photo)
background_label.image = bg_photo  # giữ tham chiếu tránh bị GC
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="dark", width=220)
menu_frame.pack(side="left", fill="y")

# Tiêu đề menu
title_label = tb.Label(
    menu_frame, 
    text="⚙️ MENU CHÍNH", 
    bootstyle="inverse-dark",
    font=("Segoe UI", 13, "bold"),
    anchor="center",
    padding=10
)
title_label.pack(fill="x", pady=(10, 20))

# ===== KHU HIỂN THỊ KẾT QUẢ =====
lbl = tb.Label(root, borderwidth=2, relief="ridge")
lbl.place_forget() 
# ===== THANH TRẠNG THÁI =====
status = tb.Label(
    root,
    text="Sẵn sàng 🚀",
    anchor="w",
    bootstyle="info",
    font=("Segoe UI", 10, "italic"),
    padding=5
)
status.pack(side="bottom", fill="x")

# ===== CÁC BIẾN TOÀN CỤC =====
cap = None
frame_count = 0
running_mode = None
after_id = None


def update_status(msg):
    status.config(text=msg)

# ===== Nhận diện ảnh =====
def detect_image_gui():
    global cap
    if cap: cap.release()

    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        output_path = detect_single_image(file_path)
        if output_path is None:
            update_status(" Không đọc được ảnh")
            return
    except Exception as e:
        update_status(f"Lỗi khi nhận diện ảnh: {e}")
        return

    # --- Hiển thị ảnh kết quả lên giao diện ---
    img = Image.open(output_path).resize((Config.IMAGE_W, Config.IMAGE_H))
    imgtk = ImageTk.PhotoImage(img)
    lbl.place(x=220, y=10, relwidth=0.7, relheight=0.9)
    lbl.config(image=imgtk)
    lbl.image = imgtk

    update_status(f"Ảnh: {os.path.basename(file_path)} | Đã lưu: {output_path}")

# ===== Dừng camera/video =====
def stop_current():
    global cap, running_mode, after_id
    running_mode = None
    if after_id:
        root.after_cancel(after_id)
    if cap and cap.isOpened():
        cap.release()
    cap = None
    update_status("Đã dừng video/camera.")

# ===== Nhận diện video =====
def detect_video():
    stop_current()
    global cap, frame_count, running_mode
    running_mode = "video"
    cap = cv2.VideoCapture(filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")]))
    if not cap or not cap.isOpened():
        update_status("Không thể mở video.")
        return
    frame_count = 0
    update_status("Đang phát video...")
    process_stream()

# ===== Nhận diện camera =====
def detect_camera():
    stop_current()
    global cap, running_mode
    running_mode = "camera"
    cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        update_status("Không thể mở camera.")
        return
    update_status("Camera bật.")
    process_stream()

# ===== Vòng lặp xử lý =====
def process_stream():
    global cap, frame_count, running_mode, after_id
    if cap is None or running_mode not in ("video", "camera"):
        return

    ret, frame = cap.read()
    if not ret:
        update_status("Kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1

    if frame_count == 1:
        lbl.place(x=250, y=40, relwidth=0.7, relheight=0.85)

    if frame_count % 5 == 0:
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((Config.IMAGE_W, Config.IMAGE_H))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        update_status(f"{running_mode.upper()} | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")

    after_id = root.after(Config.FRAME_DELAY, process_stream)


# ===== Nút chức năng =====
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image_gui, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="⏹ Dừng", bootstyle=SECONDARY, command=stop_current, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

root.mainloop()
