# Object Detection System

Real-time object detection application using YOLOv8 and webcam input, built with Python. Features a desktop GUI and automatic detection history logging.

---

## Features

- Real-time object detection via webcam using YOLOv8
- Clean desktop GUI built with Tkinter & ttkbootstrap
- Detection history automatically saved to CSV and JSON
- Fast inference with YOLOv8n (nano) model

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Detection Model | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| GUI Framework | Tkinter, ttkbootstrap 1.10.1 |
| Image Processing | Pillow |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/linhmylemai/Detection_python.git
cd Detection_python
```

**2. Install dependencies**
```bash
pip install pillow opencv-python
pip install ultralytics
pip install ttkbootstrap==1.10.1
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
python -m object_detection.app
```

## Project Structure

```
Detection_python/
    object_detection/
        app.py              # Main controller, manages GUI and modules
        camera.py           # Live camera stream handler
        video_detector.py   # Video file handler with Pause/Resume
        image_detector.py   # Static image handler
        history_view.py     # Detection history display
        config.py           # System config (paths, YOLO params, UI colors)
    docs/                   # Documentation
    tests/                  # Unit tests
    detection_log.csv       # Detection history (CSV)
    detection_log.json      # Detection history (JSON)
    yolov8n.pt              # YOLOv8 nano model weights
    requirements.txt
    README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/linhmylemai/Detection_python.git
cd Detection_python
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pillow opencv-python
pip install ultralytics
pip install ttkbootstrap==1.10.1
```

---

## Run the App

```bash
python -m object_detection.app
```

---

## Team

| MSSV | Name | Role |
|------|------|------|
| 3122410210 | Mai Le My Linh | Team Leader, UI Design, Detection History, Report |
| 3121410249 | Tran Quang Khai | Camera Module, GUI Integration |
| 3122410461 | Nguyen Huynh Phuong Uyen | Image Mode, GUI Integration |
| 3122560006 | Tra Hoang Hong Chau | Video Mode, GUI Integration |

---

## Instructor

Le Tan Long — Saigon University, Faculty of Information Technology

---

## License

This project is for educational purposes — Saigon University, 2025.
