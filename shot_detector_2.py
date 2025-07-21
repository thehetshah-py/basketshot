# everything form shot_detector
# - it detects person's position: left, middle and right
# - it stores the video output to /results

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device


class ShotDetector:
    def __init__(self):
        self.overlay_text = "Waiting..."
        self.model = YOLO("best.pt")

        self.class_names = ['Basketball', 'Basketball Hoop', 'Person']
        self.device = get_device()

        self.cap = cv2.VideoCapture("input/basket.mp4")
        
        # Get video properties
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Create VideoWriter to save output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/result_{timestamp}.mp4"

        self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height)) #save file with timestamp


        self.ball_pos = [] 
        self.hoop_pos = []
        self.person_pos = []  # ✅ NEW: store player positions

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                break

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100

                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if current_class == "Person" and conf > 0.4:
                        self.person_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 255, 0))

                        # Determine LEFT/MIDDLE/RIGHT
                        frame_width = self.frame.shape[1]
                        if center[0] < frame_width / 3:
                            position = "LEFT"
                        elif center[0] < 2 * frame_width / 3:
                            position = "MIDDLE"
                        else:
                            position = "RIGHT"

                        cv2.putText(self.frame, position, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (255, 255, 0), 2)

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1
            
            self.out.write(self.frame)

            cv2.imshow('Frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        
        cv2.destroyAllWindows()
        self.out.release()

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.overlay_text = "basket made"
                    else:
                        self.overlay_color = (255, 0, 0)
                        self.overlay_text = "basket miss"

    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if hasattr(self, 'overlay_text'):
            font_scale = 1.2
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                           font_scale, thickness)
            text_x = (self.frame.shape[1] - text_width) // 2
            text_y = 50

            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, self.overlay_color, thickness)


if __name__ == "__main__":
    ShotDetector()
