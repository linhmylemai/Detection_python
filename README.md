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

---

## Project Structure
Detection_python/
├── object_detection/     # Main application module
├── docs/                 # Documentation
├── tests/                # Unit tests
├── detection_log.csv     # Detection history (CSV)
├── detection_log.json    # Detection history (JSON)
├── yolov8n.pt            # YOLOv8 nano model weights
├── requirements.txt      # Dependencies
└── README.md
---

## Team

| Role | Name |
|------|------|
| Team Leader / UI / Camera Detection / History Log | Mai Le My Linh |

---

## License

This project is for educational purposes.
