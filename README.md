# 🏀 AI Basketball Shot Detection Tracker

This project uses YOLOv8 and computer vision techniques to detect basketball shots, passes, and made baskets from video footage.

## 🔧 Features
- Real-time shot and pass detection using YOLOv8
- Basket made detection with visual feedback
- OCR-based jersey number recognition (EasyOCR)
- ByteTrack object tracking for consistent player/ball identification

## 📁 Project Structure
- `main.py`: Entry point for running the detection pipeline
- `shot_detector.py`: Core logic for detecting shot and basket events
- `utils.py`: Helper functions
- `video/`: Store your input `.mp4` game footage here (this folder is ignored in Git)

## 🧠 Model
- Uses pretrained YOLOv8 model (`best.pt`)
- OCR via EasyOCR for jersey number tracking

## 🛠 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt