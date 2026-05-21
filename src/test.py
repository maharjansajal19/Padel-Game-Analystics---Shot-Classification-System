# Complete pipeline

import cv2
import pandas as pd
import json
import os
from detector import ObjectDetector
from pose_estimator import PoseEstimator
from tracker import BallTracker
from shot_classifier import ShotClassifier
from visualizer import Visualizer

VIDEO_PATH = '../Data/input_sample_video.mp4'
OUTPUT_VIDEO = '../Outputs/infernce_sample_video.mp4'
PROCESS_EVERY_N_FRAMES = 3

os.makedirs('../Outputs', exist_ok=True)

def main():
    detector = ObjectDetector('yolov8m.pt') # medium model handles both players + ball
    pose_estimator = PoseEstimator()
    tracker = BallTracker()
    classifier = ShotClassifier()
    visualizer = Visualizer()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Video opened:", cap.isOpened())
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )
    
    shot_results = []
    
    frame_id = 0
    
    # Cache
    last_players   = []
    last_ball      = None
    last_shot_type = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                # Detection
                players, ball = detector.detect(frame)
            
            
                # Pose estimation
                landmarks = pose_estimator.estimate(frame)
                        
                # Ball tracking
                tracker.update(ball)
            
                velocity = tracker.get_velocity()
                direction_change = tracker.get_direction_change()
                
                shot_type = None
            
                # Shot event detection
                if velocity > 1 or direction_change > 5:
                    shot_type = classifier.classify(
                        landmarks,
                        velocity,
                        direction_change
                    )
                
                if shot_type is not None:
                    
                    timestamp = frame_id/fps
                    shot_results.append({
                        'frame': frame_id,
                        'timestamp': round(timestamp, 2),
                        'shot_type': shot_type
                    })
                    
                # Update cache
                last_players   = players
                last_ball      = ball
                last_shot_type = shot_type
                
            # Visualization
            frame = visualizer.draw_players(frame, last_players)
            frame = visualizer.draw_ball(frame, last_ball)
            
            if last_shot_type is not None:
                frame = visualizer.draw_shot(frame, last_shot_type)
                
            out.write(frame)
            
            cv2.imshow('Padel Analytics', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_id += 1
        
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        with open('../Outputs/result.json', 'w') as f:
            json.dump(shot_results, f, indent=4)

        df = pd.DataFrame(shot_results) if shot_results else pd.DataFrame(columns=['frame','timestamp','shot_type'])
        df.to_csv('../Outputs/result.csv', index=False)

        print(f'Processing Complete — {len(shot_results)} shots detected')
    
if __name__ == '__main__':
    main()