import cv2
import os

# Paths
video_path = "video/basket5.mp4"
output_dir = "auto_dataset/images/train"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
frame_count = 0

print("Extracting frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every 5th frame
    if frame_count % 5 == 0:
        filename = f"frame_{frame_count}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)

    frame_count += 1

cap.release()
print("Done! Frames saved to auto_dataset/images/train/")
