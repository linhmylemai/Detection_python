import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time
from camera import CameraHandler
from config import Config

# Load YOLO model
model = YOLO(Config.MODEL_PATH)

# Khởi tạo window
root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

# ========== ẢNH NỀN ==========
bg_image = Image.open("Background.png").resize((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tb.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # full background

# ========== MENU DỌC ==========
menu_frame = tb.Frame(root, bootstyle="secondary", width=Config.MENU_WIDTH)
menu_frame.pack(side="left", fill="y")

# ========== KHU VỰC HIỂN THỊ KẾT QUẢ ==========
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=Config.MENU_WIDTH, y=0, relwidth=0.8, relheight=0.95)

# Set ảnh mặc định để tránh nền trắng
default_img = Image.open("Background.png").resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
default_photo = ImageTk.PhotoImage(default_img)
lbl.config(image=default_photo)
lbl.image = default_photo

# Status bar
status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

camera_handler = CameraHandler()

def update_status(msg):
    status.config(text=msg)

# ========== XỬ LÝ ẢNH ==========
def detect_image():
    global camera_handler
    if camera_handler.cap:
        camera_handler.release()
    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*"+Config.IMAGE_EXTENSIONS[0]+";*"+Config.IMAGE_EXTENSIONS[1]+";*"+Config.IMAGE_EXTENSIONS[2])])
    if not file_path: return
    results = model(file_path, save=True, project=Config.OUTPUT_DIR, name=Config.RESULTS_SUBDIR, exist_ok=True)
    output_file = os.path.join(Config.OUTPUT_DIR, Config.RESULTS_SUBDIR, os.path.basename(file_path))
    img = Image.open(output_file).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
    imgtk = ImageTk.PhotoImage(img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    # Tính phần trăm người và vật
    total_objects = len(results[0].boxes)
    if total_objects > 0:
        person_count = sum(1 for cls_id in results[0].boxes.cls if results[0].names[int(cls_id)] == "person")
        object_count = total_objects - person_count
        person_percent = (person_count / total_objects) * 100
        object_percent = (object_count / total_objects) * 100
        update_status(f"Ảnh: {os.path.basename(file_path)} | Người: {person_percent:.1f}% | Vật: {object_percent:.1f}% | Tổng: {total_objects}")
    else:
        update_status(f"Ảnh: {os.path.basename(file_path)} | Không phát hiện đối tượng")

# ========== XỬ LÝ VIDEO ==========
def detect_video():
    global camera_handler
    if camera_handler.cap:
        camera_handler.release()
    file_path = filedialog.askopenfilename(filetypes=[("Video", "*"+Config.VIDEO_EXTENSIONS[0]+";*"+Config.VIDEO_EXTENSIONS[1]+";*"+Config.VIDEO_EXTENSIONS[2])])
    if not file_path: return
    camera_handler.start_camera()  # Sử dụng camera tạm thời để lấy frame, sau đó thay bằng video
    cap = cv2.VideoCapture(file_path)
    camera_handler.cap = cap
    update_status(f"Video: {os.path.basename(file_path)}")
    show_frame()

# ========== XỬ LÝ CAMERA ==========
def detect_camera():
    global camera_handler
    if camera_handler.cap:
        camera_handler.release()
    camera_handler.start_camera()
    update_status("Camera bật")
    show_frame()

def show_frame():
    global camera_handler
    if not camera_handler.is_running or not camera_handler.cap:
        return
    ret, frame = camera_handler.cap.read()
    if ret:
        start = time.time()
        results = model(frame)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        # Tính phần trăm người và vật
        total_objects = len(results[0].boxes)
        if total_objects > 0:
            person_count = sum(1 for cls_id in results[0].boxes.cls if results[0].names[int(cls_id)] == "person")
            object_count = total_objects - person_count
            person_percent = (person_count / total_objects) * 100
            object_percent = (object_count / total_objects) * 100
            update_status(f"Camera/Video | FPS: {fps:.2f} | Người: {person_percent:.1f}% | Vật: {object_percent:.1f}% | Tổng: {total_objects}")
        else:
            update_status(f"Camera/Video | FPS: {fps:.2f} | Không phát hiện đối tượng")
    lbl.after(Config.FRAME_DELAY, show_frame)

# ========== NÚT TRONG MENU ==========
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

# ========== CHẠY APP ==========
root.mainloop()
