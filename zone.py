import cv2
import json
import numpy as np
from pathlib import Path

path = 'video/test.mp4'
result_file = 'restricted_zones.json'

class ZoneDrawer:
    def __init__(self, video_path, output_file):
        self.video_path = video_path
        self.output_file = output_file
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added points: {x}, {y}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        if not ret:
            print("Something is wrong with video or path")
            return

        cv2.namedWindow("Drawing Zone")
        cv2.setMouseCallback("Drawing Zone", self.mouse_callback)

        print("Instruction")
        print("Press s to save and exit")
        print("Press q to quit")

        while True:
            temp_frame = frame.copy()

            if len(self.points) > 1:
                cv2.polylines(temp_frame, [np.array(self.points)], isClosed=True, color=(0, 0, 255), thickness=2)
            
            for point in self.points:
                cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)

            cv2.imshow("Drawing Zone", temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_zone()
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_zone(self):            
        with open(self.output_file, 'w') as f:
            json.dump(self.points, f)
        print(f"Cords saved at {self.output_file}")

if __name__ == "__main__":
    drawer = ZoneDrawer(path, result_file)
    drawer.run()