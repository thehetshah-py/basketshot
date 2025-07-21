# 🏀 AI Basketball Shot Detection Tracker

This project uses YOLOv8 to detect basketballs, hoops, and players in a video, track shot attempts, classify them as make or miss, and optionally label player positions as LEFT / MIDDLE / RIGHT. It also supports saving annotated output video.

---

## 📁 Project Structure

```
AI-BASKETBALL-SHOT-DETECTION-TRACKER/
├── auto_dataset/               ← training images + labels (ignored in Git)
├── backups/                    ← backup of old best.pt models
├── output/                     ← saved result videos (e.g., result_20250720_1845.mp4)
├── video/                      ← input videos (e.g., basket5.mp4)
├── best.pt                     ← trained YOLOv8 model (not committed)
├── config.yaml                 ← YOLO dataset configuration
├── main.py                     ← model training script
├── shot_detector.py            ← detects ball, hoop, person
├── shot_detector_2.py          ← adds player position tagging + video saving
├── utils.py                    ← detection logic, filtering, scoring
├── extract_frames.py           ← convert video to frames
├── auto_label_person.py        ← auto-labels players using pretrained YOLOv8
├── add_ball_and_hoop_labels.py ← runs best.pt to label ball & hoop
├── yolov8n.pt                  ← pretrained COCO model for person detection
└── requirements.txt            ← Python dependencies (optional)
```

---

## 🚀 Features

* Detects **basketball**, **hoop**, and **player**
* Tracks **ball movement** to detect shot attempts
* Determines if a shot is **made or missed**
* Labels each player as **LEFT / MIDDLE / RIGHT**
* Saves output video with annotations and score overlay

---

## 🧐 How It Works

1. YOLOv8 detects objects per frame.
2. The ball’s motion is tracked across the hoop region.
3. If a ball moves from above the hoop to below, it’s considered an **attempt**.
4. If the path intersects the hoop region, it's marked a **make**.
5. Each detected person is tagged by screen position (left/middle/right).
6. The result is displayed on screen and optionally saved to a video file.

---

## 🛠️ Setup

### Step 1: Clone the Repo

```bash
git clone https://github.com/your-username/AI-BASKETBALL-SHOT-DETECTION-TRACKER.git
cd AI-BASKETBALL-SHOT-DETECTION-TRACKER
```

### Step 2: Install Python Dependencies

```bash
pip install ultralytics opencv-python cvzone numpy
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## 🏋️️ Training the Model from Scratch

You'll train a YOLOv8 model to detect:

* `Basketball` → class `0`
* `Basketball Hoop` → class `1`
* `Person` → class `2`

### ✅ Step-by-Step

#### 1. Extract Frames from Video

```bash
python extract_frames.py
```

#### 2. Auto-Label Persons Using Pretrained Model

```bash
python auto_label_person.py
```

#### 3. Add Labels for Basketball and Hoop

```bash
python add_ball_and_hoop_labels.py
```

#### 4. Edit the Dataset Config (`config.yaml`)

```yaml
train: auto_dataset/images/train
val: auto_dataset/images/train

nc: 3
names: ['Basketball', 'Basketball Hoop', 'Person']
```

#### 5. Train the YOLOv8 Model

```bash
python main.py
```

💡 After training, your model is saved at:

```
runs/detect/train/weights/best.pt
```

Copy it to the project root as `best.pt`.

---

## 🎯 Running the Detectors

### ▶️ `shot_detector.py` – Basic Version

* Detects ball, hoop, and person
* Displays live score and bounding boxes

```bash
python shot_detector.py
```

### ▶️ `shot_detector_2.py` – Extended Version

* Also tags each player as LEFT, MIDDLE, or RIGHT
* Saves output video to `output/` with timestamped filename

```bash
python shot_detector_2.py
```

📝 Example output:

```
output/result_20250720_1845.mp4
```

---

## 💾 Output Video Saving

The extended version automatically saves annotated video with:

* Score
* Shot result (make/miss)
* Player location labels

You can find all saved files in the `output/` folder.

---

## 📆 .gitignore

To avoid committing large files or generated artifacts, your `.gitignore` should include:

```gitignore
# Output videos
output/
*.mp4

# Model weights
*.pt

# Dataset files
auto_dataset/
*.zip

# Cache
__pycache__/
*.pyc

# Backups
backups/

# System files
.DS_Store
```

---

## 📜 Notes

* Place your input video inside the `video/` folder (e.g., `video/basket5.mp4`).
* Scripts like `auto_label_person.py` are only used for dataset generation (optional).
* You can extend the system to recognize jersey numbers, detect passes, or track players.

---

## 📌 Credits

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/) + [cvzone](https://github.com/cvzone/cvzone)

---

## ✅ Status

* ✅ Training Pipeline (YOLOv8)
* ✅ Shot Detection (ball & hoop logic)
* ✅ Player Detection & Position Tagging
* ✅ Output Video Saving with Timestamp
