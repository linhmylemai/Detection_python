import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time
import json

# Load YOLO model
model = YOLO("yolov8n.pt")

root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry("1100x700")

# ===== ẢNH NỀN =====
bg_image = Image.open("Background.png").resize((1100, 700))
bg_photo = ImageTk.PhotoImage(bg_image)
tb.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="secondary", width=200)
menu_frame.pack(side="left", fill="y")

# ===== KHU HIỂN THỊ KẾT QUẢ =====
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=200, y=0, relwidth=0.8, relheight=0.95)
default_img = Image.open("Background.png").resize((850, 600))
default_photo = ImageTk.PhotoImage(default_img)
lbl.config(image=default_photo)
lbl.image = default_photo

status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

cap = None
frame_count = 0
running_mode = None
after_id = None  # để hủy vòng lặp Tkinter

# Load config
from config import CONFIG
LOG_FILE = "detection_log.json"

def save_detection_data(file_path, object_count):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": os.path.basename(file_path) if file_path else "Live Camera",
        "objects_detected": object_count
    }
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
    except Exception as e:
        update_status("❌ Lỗi khi lưu log: " + str(e))

def update_status(msg):
    status.config(text=msg)

def stop_current():
    """Dừng tất cả video/camera đang chạy."""
    global cap, running_mode, after_id
    running_mode = None
    if after_id:
        root.after_cancel(after_id)
    if cap and cap.isOpened():
        cap.release()
    cap = None
    update_status("⏹ Đã dừng video/camera.")
    lbl.config(image=default_photo)
    lbl.image = default_photo

# ===== XỬ LÝ ẢNH =====
def detect_image():
    stop_current()
    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    results = model(file_path)
    img = results[0].plot()
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((850, 600))
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    save_detection_data(file_path, len(results[0].boxes))
    update_status(f"Ảnh: {os.path.basename(file_path)} | Đối tượng: {len(results[0].boxes)}")

# ===== XỬ LÝ VIDEO =====
def detect_video():
    stop_current()
    global cap, frame_count, running_mode
    running_mode = "video"
    cap = cv2.VideoCapture(filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")]))
    if not cap or not cap.isOpened():
        update_status("❌ Không thể mở video.")
        return
    frame_count = 0
    update_status("🎥 Đang phát video...")
    process_stream()

# ===== XỬ LÝ CAMERA =====
def detect_camera():
    stop_current()
    global cap, running_mode
    running_mode = "camera"
    cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        update_status("❌ Không thể mở camera.")
        return
    update_status("📡 Camera bật.")
    process_stream()

# ===== VÒNG LẶP XỬ LÝ CHUNG =====
def process_stream():
    global cap, frame_count, running_mode, after_id
    if cap is None or running_mode not in ("video", "camera"):
        return
    ret, frame = cap.read()
    if not ret:
        update_status("📁 Kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1
    if frame_count % 5 == 0:  # skip frame để nhẹ CPU
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        save_detection_data(None, len(results[0].boxes))
        update_status(f"{running_mode.upper()} | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")

    after_id = root.after(20, process_stream)

# ===== NÚT MENU =====
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="⏹ Dừng", bootstyle=SECONDARY, command=stop_current, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

root.mainloop()
