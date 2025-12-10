import cv2
import json
import time
import torch
import numpy as np
import os
from ultralytics import YOLO

video_path = 'video/test.mp4'
zone_path = 'restricted_zones.json'
model_name = 'yolov8m.pt'

confidence_treshold = 0.45
cooldown = 3.0

tracker_content = """
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.2
new_track_thresh: 0.5
track_buffer: 2500
match_thresh: 0.7
fuse_score: True
"""

tracker_file = "tracker.yaml"
with open(tracker_file,"w") as f:
    f.write(tracker_content)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB-xA)
    interH = max(0, yB-yA)
    interArea = interW*interH
    if interArea == 0: return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea/float(boxAArea+boxBArea-interArea)

def crop_safe(img, bbox):
    h, w = img.shape[:2]
    x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
    x2 = min(w-1, bbox[2]); y2 = min(h-1, bbox[3])
    if x2<=x1 or y2<=y1:
        return None
    return img[y1:y2, x1:x2]

def hsv_hist(image):
    if image is None or image.size==0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
    cv2.normalize(h, h)
    return h

class SecuritySystem:
    def __init__(self, video_path, zone_file):
        self.cap = cv2.VideoCapture(video_path)
        self.zone = self.load_zone(zone_file)
        self.model = YOLO(model_name)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.alarm_active = False
        self.last_intrusion_time = 0
        self.intrusion_frames = 0
        self.required_frames = 5
        self.lost_cache = {}
        self.cache_ttl = 3.0
        self.frame_time = time.time()

    def load_zone(self,path):
        with open(path,'r') as f:
            pts=json.load(f)
        return np.array(pts,dtype=np.int32).reshape((-1,1,2))

    def cleanup_cache(self):
        now = time.time()
        remove = [k for k,v in self.lost_cache.items() if now - v['last_seen'] > self.cache_ttl]
        for k in remove:
            del self.lost_cache[k]

    def process_frame(self, frame):
        H, W = frame.shape[:2]
        dynamic_min_feet_y = int(H*0.50)
        dynamic_max_feet_y = int(H*0.98)

        results = self.model.track(frame, persist=True, tracker=tracker_file, classes=[0],
                                   conf=confidence_treshold, iou=0.5, imgsz=640,
                                   device=self.device, verbose=False)

        intruder_present = False
        seen_ids = set()
        current_time = time.time()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            displayed_ids = []
            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                feet_y = y2
                cx = (x1+x2)//2
                if feet_y < dynamic_min_feet_y:
                    continue

                patch = crop_safe(frame, (x1,y1,x2,y2))
                hist = hsv_hist(patch)

                matched_old_id = None
                best_score = 0.0
                for old_id,info in list(self.lost_cache.items()):
                    if current_time-info['last_seen'] > self.cache_ttl:
                        continue
                    iou_score = iou((x1,y1,x2,y2), info['bbox'])
                    dist = np.hypot(cx-info['center'][0], feet_y-info['center'][1])
                    hist_score = 0.0
                    if hist is not None and info['hist'] is not None:
                        hist_score = cv2.compareHist(hist, info['hist'], cv2.HISTCMP_CORREL)
                    combined = (iou_score*0.6)+(hist_score*0.3)+((1-min(dist/300.0,1.0))*0.1)
                    if combined > best_score and (iou_score>0.20 or hist_score>0.35 or dist<120):
                        best_score = combined
                        matched_old_id = old_id

                if matched_old_id is not None and matched_old_id not in seen_ids:
                    display_id = matched_old_id
                    del self.lost_cache[matched_old_id]
                else:
                    display_id = int(obj_id)

                seen_ids.add(display_id)
                self.lost_cache[display_id] = {'bbox':(x1,y1,x2,y2),'center':(cx,feet_y),'hist':hist,'last_seen':current_time}

                inside_zone = cv2.pointPolygonTest(self.zone, (float(cx), float(feet_y)), False) >= 0

                if inside_zone:
                    intruder_present = True
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{display_id}", (x1, max(25, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        to_remove = []
        for old_id,info in list(self.lost_cache.items()):
            if current_time - info['last_seen'] > self.cache_ttl:
                to_remove.append(old_id)
        for k in to_remove:
            del self.lost_cache[k]

        if intruder_present:
            self.intrusion_frames += 1
            if self.intrusion_frames >= self.required_frames:
                self.last_intrusion_time = time.time()
                self.alarm_active = True
        else:
            self.intrusion_frames = 0
            if time.time()-self.last_intrusion_time > cooldown:
                self.alarm_active = False

        return frame

    def draw_hud(self,frame):
        if len(self.zone)>0:
            cv2.polylines(frame,[self.zone],True,(0,140,255),2)
        txt="ALARM" if self.alarm_active else "SECURE"
        col=(0,0,255) if self.alarm_active else (0,255,0)
        cv2.putText(frame,txt,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,3)
        return frame

    def run(self):
        cv2.namedWindow("Surveillance",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Surveillance",1280,720)
        while True:
            ret,frame=self.cap.read()
            if not ret: break
            out=self.process_frame(frame)
            out=self.draw_hud(out)
            cv2.imshow("Surveillance",out)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        self.cap.release()
        cv2.destroyAllWindows()
        try:
            os.remove(tracker_file)
        except:
            pass

if __name__=="__main__":
    SecuritySystem(video_path,zone_path).run()