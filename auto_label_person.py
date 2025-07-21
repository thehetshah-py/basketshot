from ultralytics import YOLO
import os

# Load pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for better accuracy if desired

# Paths
image_dir = "auto_dataset/images/train"
label_dir = "auto_dataset/labels/train"
os.makedirs(label_dir, exist_ok=True)

# Person class in COCO is 0, but in our dataset we map it to 2
COCO_PERSON_ID = 0
CUSTOM_PERSON_ID = 2

print("Auto-labeling person class...")

# Process each image
for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, image_file)
    results = model(image_path)

    label_lines = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == COCO_PERSON_ID:
            # YOLO format: class x_center y_center width height (normalized)
            x_center, y_center, width, height = box.xywh[0]
            img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

            x_center /= img_w
            y_center /= img_h
            width /= img_w
            height /= img_h

            label_line = f"{CUSTOM_PERSON_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            label_lines.append(label_line)

    # Save label file
    if label_lines:
        label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

print("Auto-labeling completed. Labels saved to auto_dataset/labels/train/")
