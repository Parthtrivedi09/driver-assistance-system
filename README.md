# 🚗 Driver Assistance System using Deep Learning

## 📖 Overview
This project builds a real-time driver assistance system that detects road signs (speed limits, school zones, stop signs) using computer vision and alerts the driver.

## 🚀 Features
- Real-time traffic sign detection
- Speed limit recognition
- Driver alert system (visual + optional audio)

## 🧠 Tech Stack
- YOLOv8
- OpenCV
- PyTorch

## 📂 Project Structure
(Explain folders)

## 🎯 Future Work
- Distance estimation
- Voice alerts
- Integration with GPS


## 📊 Project Progress
- [x] Project setup 
- [ ] YOLO detection
- [ ] Custom training
- [ ] Real-time system
- [ ] Driver alerts


# annotation file 
What annotation file looks like:

Example:

00000.ppm;774;411;815;446;11
🧠 Meaning:
Field	Meaning
filename	image name
x1, y1	top-left corner
x2, y2	bottom-right
class	sign type


Mid pipeline 
Camera / Video Feed
        ↓
Frame Capture
        ↓
YOLO Detection
        ↓
Filter Only STOP Sign
        ↓
Confidence Check
        ↓
Temporal Validation
        ↓
Trigger Alert
        ↓
Voice / Warning Display