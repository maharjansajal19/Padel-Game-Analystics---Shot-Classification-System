'''
Performs:
    - YOLO object detection
    - Finds players
    - Finds the ball
    - Finds rackets
'''

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path = 'yolov8m.pt'):
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        
        # Pass 1 — players at fast resolution
        results_players = self.model(
            frame, conf=0.3, iou=0.4, imgsz=640, verbose=False, device=0
        )[0]

        # Pass 2 — ball at high resolution
        results_ball = self.model(
            frame, conf=0.1, iou=0.4, imgsz=1280, verbose=False, device=0
        )[0]

        
        players = []
        ball = None
        rackets = []
        best_ball_conf = 0
        
        for box in results_players.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = results_players.names[cls]
            
            # Detect players
            if label == 'person' and conf > 0.3:
                w, h = x2 - x1, y2 - y1
                if w * h > 500:
                    players.append({'bbox': (x1, y1, x2, y2), 'confidence': conf})
                    
            if label == 'tennis racket' and conf > 0.3:
                rackets.append({'bbox': (x1, y1, x2, y2), 'confidence': conf})

        # Detect sports ball
        for box in results_ball.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results_ball.names[cls]
            if label == 'sports ball' and conf > 0.1:
                if conf > best_ball_conf:
                    best_ball_conf = conf
                    ball = {'bbox': (x1, y1, x2, y2), 'confidence': conf}
        
        return players, ball, rackets