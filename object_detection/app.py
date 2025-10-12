import os
import time
import shutil
import cv2
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from detectors.image_detector import detect_single_image
from config import Config

# ===== Khởi tạo thư mục =====
os.makedirs(Config.INPUTS, exist_ok=True)
os.makedirs(Config.OUTPUTS, exist_ok=True)

# ===== Nạp mô hình =====
model = YOLO(os.path.join(Config.MODELS_DIR, Config.MODEL_PATH))

# ===== Giao diện chính =====
root = tb.Window(themename="cosmo")
root.title("Object Detection App")
root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

# ===== Ảnh nền =====
bg_path = os.path.join(Config.BASE_DIR, "assets", "Background.png")
bg_image = Image.open(bg_path).resize(
    (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
)
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tb.Label(root, image=bg_photo)
background_label.image = bg_photo  # giữ tham chiếu tránh bị GC
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="dark", width=180)
menu_frame.pack(side="left", fill="y")

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

# ===== KHUNG CHỨA VIDEO (Pause / Replay) =====
video_control_frame = tb.Frame(root, bootstyle="dark")
video_control_frame.place_forget()  # Ẩn khi chưa chọn video

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
paused = False


# ===== Hàm cập nhật trạng thái =====
def update_status(msg):
    status.config(text=msg)


# ===== Nhận diện ảnh =====
def detect_image_gui():
    global cap
    stop_current()
    video_control_frame.place_forget()  

    file_path = filedialog.askopenfilename(
        filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    try:
        output_path = detect_single_image(file_path)
        if output_path is None:
            update_status("Không đọc được ảnh.")
            return
    except Exception as e:
        update_status(f"Lỗi khi nhận diện ảnh: {e}")
        return

    img = Image.open(output_path).resize(
        (Config.IMAGE_W, Config.IMAGE_H)
    )
    imgtk = ImageTk.PhotoImage(img)
    lbl.place(x=280, y=15, relwidth=0.7, relheight=0.9)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    update_status(f"Ảnh: {os.path.basename(file_path)} | Đã lưu: {output_path}")


# ===== Dừng video / camera =====
def stop_current():
    global cap, running_mode, after_id
    running_mode = None
    if after_id:
        root.after_cancel(after_id)
    if cap and cap.isOpened():
        cap.release()
    cap = None
    video_control_frame.place_forget()
    update_status("⏹ Đã dừng video/camera.")


# ===== Tạm dừng / tiếp tục =====
def toggle_pause():
    global paused
    paused = not paused
    if paused:
        update_status("⏸ Video tạm dừng.")
        btn_pause.config(text="▶ Tiếp tục", bootstyle="success-outline")
    else:
        update_status("🎬 Tiếp tục phát video...")
        btn_pause.config(text="⏸ Tạm dừng", bootstyle="warning-outline")


# ===== Phát lại video =====
def replay_video():
    global cap, running_mode, frame_count, paused
    if running_mode != "video" or cap is None:
        update_status("Không có video nào để phát lại.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    paused = False
    btn_pause.config(text="⏸ Tạm dừng", bootstyle="warning-outline")
    update_status("🔁 Phát lại video từ đầu.")


# ===== Xử lý video =====
def detect_video():
    stop_current()
    global cap, frame_count, running_mode
    running_mode = "video"

    file_path = filedialog.askopenfilename(
        filetypes=[("Video", "*.mp4;*.avi;*.mov")]
    )
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        update_status("❌ Không thể mở video.")
        return

    frame_count = 0
    update_status("🎬 Đang phát video...")

    video_control_frame.place(x=1140, y=600)
    lbl.place(x=180, y=15, relwidth=0.7, relheight=0.9)
    process_stream()


# ===== Nhận diện camera =====
def detect_camera():
    stop_current()
    video_control_frame.place_forget()  # Ẩn khung điều khiển video

    global cap, running_mode
    running_mode = "camera"
    cap = cv2.VideoCapture(0)

    if not cap or not cap.isOpened():
        update_status("❌ Không thể mở camera.")
        return

    update_status("📸 Camera bật.")
    lbl.place(x=280, y=15, relwidth=0.7, relheight=0.9)
    process_stream()


# ===== Vòng lặp xử lý =====
def process_stream():
    global cap, frame_count, running_mode, after_id, paused

    if cap is None or running_mode not in ("video", "camera"):
        return

    if paused:
        after_id = root.after(100, process_stream)
        return

    ret, frame = cap.read()
    if not ret:
        update_status("Kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1

    if frame_count % 5 == 0:
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)

        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize(
            (Config.IMAGE_W, Config.IMAGE_H)
        )
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)

        update_status(
            f"{running_mode.upper()} | FPS: {fps:.2f} | "
            f"Đối tượng: {len(results[0].boxes)}"
        )

    after_id = root.after(Config.FRAME_DELAY, process_stream)


# ===== Nút chức năng =====
tb.Button(
    menu_frame,
    text="📷 Ảnh",
    bootstyle=SUCCESS,
    command=detect_image_gui,
    width=15
).pack(pady=15)

tb.Button(
    menu_frame,
    text="🎥 Video",
    bootstyle=INFO,
    command=detect_video,
    width=15
).pack(pady=15)

tb.Button(
    menu_frame,
    text="📡 Camera",
    bootstyle=PRIMARY,
    command=detect_camera,
    width=15
).pack(pady=15)

tb.Button(
    menu_frame,
    text="❌ Thoát",
    bootstyle=DANGER,
    command=root.destroy,
    width=15
).pack(pady=15)

# ===== Nút điều khiển video =====
btn_pause = tb.Button(
    video_control_frame,
    text="⏸ Tạm dừng",
    bootstyle="warning",  # màu vàng nổi bật
    width=12,
    padding=5,
    command=toggle_pause,
)
btn_pause.pack(side="left", padx=10, pady=8)

btn_replay = tb.Button(
    video_control_frame,
    text="🔁 Phát lại",
    bootstyle="info",  # màu xanh dương nhẹ
    width=12,
    padding=5,
    command=replay_video,
)
btn_replay.pack(side="left", padx=10, pady=8)

# ===== Chạy ứng dụng =====
root.mainloop()
