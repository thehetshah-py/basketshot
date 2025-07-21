from ultralytics import YOLO
import os

# Load your custom-trained YOLOv8 model
model = YOLO("best.pt")

# Paths
image_dir = "auto_dataset/images/train"
label_dir = "auto_dataset/labels/train"

print("Adding ball and hoop labels...")

for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

    # Run detection
    results = model(image_path)

    label_lines = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            label_lines = f.read().splitlines()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls not in [0, 1]:  # Only interested in ball and hoop
            continue

        # Convert to YOLO format (normalized)
        x_center, y_center, width, height = box.xywh[0]
        img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

        x_center /= img_w
        y_center /= img_h
        width /= img_w
        height /= img_h

        label_line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        label_lines.append(label_line)

    # Save updated label file
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

print("Ball and hoop labels added to existing label files.")
