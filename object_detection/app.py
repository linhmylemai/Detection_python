import os
import threading
import time
import shutil
import cv2
from matplotlib.pyplot import box
from torch import classes
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from webcolors import names

# ✅ Import chuẩn theo package
from object_detection.detectors.image_detector import detect_single_image
from object_detection.detectors.video_detector import VideoDetector
from object_detection.detectors.camera_detector import CameraHandler
from object_detection.utils.save_log import LOG_FILE, save_detection_log
from object_detection.config import Config




# ===== Khởi tạo thư mục =====
os.makedirs(Config.INPUTS, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUTS, "results"), exist_ok=True)

# ===== Nạp mô hình =====
model = YOLO(os.path.join(Config.MODELS_DIR, Config.MODEL_PATH))
print("✅ Mô hình YOLO đã sẵn sàng!")
video_detector = VideoDetector(os.path.join(Config.MODELS_DIR, Config.MODEL_PATH))

# ===== GIAO DIỆN CHÍNH =====
root = tb.Window(themename="cosmo")
root.title("Object Detection App")
root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

# ===== ẢNH NỀN =====
bg_path = os.path.join(Config.BASE_DIR, "assets", "Background.png")
if os.path.exists(bg_path):
    bg_image = Image.open(bg_path).resize((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tb.Label(root, image=bg_photo)
    background_label.image = bg_photo
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
else:
    messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nền Background.png trong thư mục assets!")
    root.configure(bg="#1e1e1e")

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

last_status_text = ""

# ===== KHUNG CHỨA NÚT PAUSE =====
pause_frame = tb.Frame(root)
pause_frame.place_forget()
# ===== KHUNG CHỨA VIDEO (Pause / Replay) =====
video_control_frame = tb.Frame(lbl, bootstyle="dark")
video_control_frame.place_forget()

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

after_id = None
cap = None
running_mode = None

# ===== Hàm cập nhật trạng thái =====
def update_status(msg):
    global last_status_text
    # chỉ cập nhật khi nội dung khác trước đó
    if msg != last_status_text:
        status.config(text=msg)
        last_status_text = msg
    

# ===== Nhận diện ảnh =====
def detect_image_gui():
    """Chọn ảnh và nhận diện trong thread phụ (UI không bị đơ)"""
    global cap
    stop_current()
    video_control_frame.place_forget()

    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    update_status("🖼️ Đang nhận diện ảnh... Vui lòng đợi.")

    def worker():
        try:
            # Gọi YOLO trong luồng phụ
            output_path = detect_single_image(file_path, model)
            if not output_path:
                root.after(0, lambda: update_status("❌ Không thể nhận diện ảnh."))
                return

            # === Đếm số lượng đối tượng ===
            results = model(file_path, verbose=False)
            names = results[0].names
            classes = results[0].boxes.cls.tolist() if len(results[0].boxes) > 0 else []

            # Nhóm phân loại
            people_labels = ["person"]
            animal_labels = ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]
            object_labels = [n for n in names.values() if n not in people_labels + animal_labels]

            num_people = sum(1 for c in classes if names[int(c)] in people_labels)
            num_animals = sum(1 for c in classes if names[int(c)] in animal_labels)
            num_objects = sum(1 for c in classes if names[int(c)] in object_labels)

            # Hiển thị ảnh đã xử lý
            img = Image.open(output_path).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
            imgtk = ImageTk.PhotoImage(img)

            def update_ui():
                lbl.place(x=280, y=15, relwidth=0.7, relheight=0.9)
                lbl.config(image=imgtk)
                lbl.image = imgtk
                update_status(
                 f"ẢNH | Người: {num_people} | Động vật: {num_animals} | Đồ vật: {num_objects} | Đã lưu: {output_path}"
    )

              # === Ghi log nhận diện ===
                detected_objects = []
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = results[0].names[cls]
                        detected_objects.append((label, conf))

                if detected_objects:
                    save_detection_log("Ảnh", file_path, detected_objects)

            root.after(0, update_ui)

        except Exception as e:
            err_msg = str(e)
            root.after(0, lambda msg=err_msg: update_status(f"❌ Lỗi khi nhận diện: {msg}"))

    # chạy YOLO trong thread riêng
    threading.Thread(target=worker, daemon=True).start()

# ===== Dừng video / camera =====
def stop_current():
    global after_id, cap, running_mode
    video_detector.stop()
    if after_id:
        root.after_cancel(after_id)
        after_id = None
    if cap and cap.isOpened():
        cap.release()
    cap = None
    running_mode = None
    lbl.place_forget()
    video_control_frame.place_forget()
    update_status("⏹ Đã dừng video/camera.")

# ===== Xử lý video =====
def toggle_pause(btn):
    paused = video_detector.toggle_pause()
    if paused:
        update_status("⏸ Video tạm dừng.")
        btn.config(text="▶", bootstyle="success-outline")
    else:
        update_status("🎬 Tiếp tục phát video...")
        btn.config(text="⏸", bootstyle="warning-outline")
    root.after(0, process_stream)  # Đảm bảo luồng tiếp tục sau khi thay đổi trạng thái
# Trong detect_video_gui, thay đổi đoạn tạo nút và thêm toggle_pause với tham chiếu nút
def detect_video_gui():
    stop_current()
    file_path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return
    if not video_detector.detect_video(file_path):
        update_status("❌ Không thể mở video. Kiểm tra định dạng hoặc file.")
        return

    global running_mode
    running_mode = "video"
    
    update_status("🎬 Đang phát video...")
    lbl.place(x=310, y=20, width=Config.IMAGE_W, height=Config.IMAGE_H)

    global btn_pause
    btn_pause = tb.Button(root, text="⏸", bootstyle="warning", width=10, command=lambda: toggle_pause(btn_pause))
    btn_pause.place(x=310, y=20 + Config.IMAGE_H - 90, width=80, height=30)
    print(f"Button pause placed at ({310}, {20 + Config.IMAGE_H - 90}) with size 80x30")
        
    process_stream()

# ===== Nhận diện camera =====
def detect_camera():
    stop_current()
    video_control_frame.place_forget()

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
    global after_id, cap, running_mode

    if running_mode == "video":
        annotated, results, fps = video_detector.process_stream()
        if annotated is not None:
            img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk)

            # ===== PHÂN LOẠI THEO NHÓM =====
            names = results[0].names
            classes = results[0].boxes.cls.tolist() if len(results[0].boxes) > 0 else []
            
            # Danh sách nhóm
            people_labels = ["person"]
            animal_labels = ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]
            object_labels = [n for n in names.values() if n not in people_labels + animal_labels]

            # Đếm từng loại
            num_people = sum(1 for c in classes if names[int(c)] in people_labels)
            num_animals = sum(1 for c in classes if names[int(c)] in animal_labels)
            num_objects = sum(1 for c in classes if names[int(c)] in object_labels)

        # ===== HIỂN THỊ KẾT QUẢ =====
            update_status(
                f"{running_mode.upper()} | FPS: {fps:.1f} | Người: {num_people} |  Động vật: {num_animals} | Đồ vật: {num_objects}"
)
            # === Ghi log nhận diện ===
            detected_objects = []
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls]
                    detected_objects.append((label, conf))
            if detected_objects:
                save_detection_log(running_mode.capitalize(), video_detector.current_video_path or "Live Stream", detected_objects)
        else:
            update_status("✅ Video đã kết thúc. Chọn video mới để tiếp tục.")
            lbl.place_forget()
            pause_frame.place_forget()
            root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")  # Reset kích thước cửa sổ
            if after_id:
                root.after_cancel(after_id)
                after_id = None            
            return
        
    elif running_mode == "camera":
        if cap is None or not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            update_status("❌ Camera gặp lỗi.")
            return

        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)

        # Hiển thị hình ảnh
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)

        # Phân loại và hiển thị kết quả
        names = results[0].names
        classes = results[0].boxes.cls.tolist() if len(results[0].boxes) > 0 else []
        people_labels = ["person"]
        animal_labels = ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]
        object_labels = [n for n in names.values() if n not in people_labels + animal_labels]
        num_people = sum(1 for c in classes if names[int(c)] in people_labels)
        num_animals = sum(1 for c in classes if names[int(c)] in animal_labels)
        num_objects = sum(1 for c in classes if names[int(c)] in object_labels)
        update_status(
            f"CAMERA | FPS: {fps:.1f} | Người: {num_people} | Động vật: {num_animals} | Đồ vật: {num_objects}"
        )

        # Ghi log nhận diện
        detected_objects = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls]
                detected_objects.append((label, conf))
        if detected_objects:
            save_detection_log("Camera", "Live Stream", detected_objects)
    after_id = root.after(Config.FRAME_DELAY, process_stream)

# ===== MỞ CỬA SỔ LỊCH SỬ =====
import csv

def open_history_window():
    """Mở cửa sổ pastel hiển thị lịch sử nhận diện"""
    import object_detection.utils.save_log as log_file_ref
    LOG_FILE = log_file_ref.LOG_FILE
    print(f"[DEBUG] Đang đọc log từ: {LOG_FILE}")

    PASTEL_BG = "#F9FBFD"
    HEADER_BG = "#CDE8E5"
    ROW_ODD = "#E8F0F2"
    ROW_EVEN = "#F6FAFA"
    TEXT_COLOR = "#211E20"
    HIGHLIGHT = "#99CCCC"

    history_win = tb.Toplevel()
    history_win.title("📜 Lịch sử nhận diện đối tượng")
    history_win.geometry("950x500")
    history_win.configure(bg=PASTEL_BG)

    title = tb.Label(
        history_win,
        text="🗂️ Lịch sử nhận diện đối tượng",
        font=("Segoe UI", 16, "bold"),
        background=PASTEL_BG,
        foreground="#2E8B8B",
    )
    title.pack(pady=10)

    frame = tb.Frame(history_win)
    frame.pack(expand=True, fill="both", padx=15, pady=5)

    columns = ("Thời gian", "Chức năng", "Tên tệp", "Đối tượng", "Độ chính xác")
    table = tb.Treeview(frame, columns=columns, show="headings", height=15)

    for col in columns:
        table.heading(col, text=col)
        table.column(col, anchor="center", width=180)

    scroll_y = tb.Scrollbar(frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    table.pack(expand=True, fill="both")

    style = tb.Style("cosmo")
    style.configure(
        "Treeview",
        background=ROW_EVEN,
        foreground=TEXT_COLOR,
        rowheight=28,
        fieldbackground=ROW_EVEN,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        font=("Segoe UI", 11, "bold"),
        background=HEADER_BG,
        foreground=TEXT_COLOR,
    )
    style.map("Treeview", background=[("selected", HIGHLIGHT)])

    def load_data():
        for row in table.get_children():
            table.delete(row)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # bỏ header
                for i, row in enumerate(reader):
                    print("[DEBUG] Dòng log:", row)  # ✅ đúng vị trí
                    tag = "evenrow" if i % 2 == 0 else "oddrow"
                    table.insert("", "end", values=row, tags=(tag,))
        else:
            table.insert("", "end", values=("⚠️", "Chưa có dữ liệu", "", "", ""))

        table.tag_configure("oddrow", background=ROW_ODD)
        table.tag_configure("evenrow", background=ROW_EVEN)

    tb.Button(
        history_win, text="🔄 Làm mới dữ liệu", bootstyle="success", width=18, command=load_data
    ).pack(pady=8)

    load_data()

# ===== Nút MENU =====
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image_gui, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video_gui, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)
tb.Button(menu_frame, text="📜 Lịch sử", bootstyle=SECONDARY, command=open_history_window, width=15).pack(pady=15)

# ===== Chạy ứng dụng =====
root.mainloop()
