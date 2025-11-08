# AutoVision.AI – Smart Object Detection System for Autonomous Vehicles

## 🧠 Overview
AutoVision.AI is a deep learning-based project built using YOLOv8 and OpenCV to detect and classify multiple road objects like cars, pedestrians, and traffic signs in real time.

## ⚙️ Features
- Real-time object detection from webcam or video feed
- YOLOv8-based deep learning model
- Confidence-based labeling
- Modular and scalable code

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train model:
   ```bash
   python train_yolo.py --data dataset.yaml
   ```
3. Detect objects:
   ```bash
   python detect_yolo.py --weights yolov8n.pt --source 0
   ```

## 👤 Author
sneha shalagar – Week 1 Project Submission (Edunet Foundation)
