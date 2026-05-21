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
from collections import Counter
from dashboard import Dashboard

VIDEO_PATH = '../Data/infernce_sample_video.mp4'
OUTPUT_VIDEO = '../Outputs/annotated.mp4'
PROCESS_EVERY_N_FRAMES = 3

os.makedirs('../Outputs', exist_ok=True)

def main():
    detector = ObjectDetector('yolov8m.pt') # medium model handles both players + racket +  ball
    pose_estimator = PoseEstimator()
    tracker = BallTracker()
    classifier = ShotClassifier()
    visualizer = Visualizer()
    dashboard = Dashboard(width = 400)
    
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
        (width + 400, height)
    )
    
    shot_results = []
    
    frame_id = 0
    
    # Cache
    last_players   = []
    last_rackets = []
    last_ball      = None
    last_shot_type = None
    last_velocity = 0
    last_direction_change = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                # Detection
                players, ball, rackets = detector.detect(frame)
            
            
                # Pose estimation
                ##landmarks = pose_estimator.estimate(frame)
                if players:
                    largest = max(players, key=lambda p:
                        (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1]))
                    x1, y1, x2, y2 = largest['bbox']
                    pad  = 40
                    crop = frame[max(0,y1-pad):min(height,y2+pad),
                                 max(0,x1-pad):min(width,x2+pad)]
                    landmarks = pose_estimator.estimate(crop) if crop.size > 0 else {}
                else:
                    landmarks = {}       
                    
                # Ball tracking
                tracker.update(ball)
            
                velocity = tracker.get_velocity()
                direction_change = tracker.get_direction_change()
                
                shot_type = None
            
                # Shot event detection
                if velocity > 5 or direction_change > 10:
                    shot_type = classifier.classify(
                        landmarks,
                        velocity,
                        direction_change
                    )
                
                direction = classifier.get_shot_direction(tracker.get_trail())
                
                if shot_type is not None:
                    
                    timestamp = frame_id/fps
                    shot_results.append({
                        'frame': frame_id,
                        'timestamp': round(timestamp, 2),
                        'shot_type': shot_type,
                        'direction': direction
                    })
                    
                
                bounce = tracker.detect_bounce()
                
                if bounce:
                    shot_results.append({
                        'frame': frame_id,
                        'timestamp': round(frame_id / fps, 2),
                        'shot_type': 'Bounce',
                        'direction': direction
                    })
                
                # Update cache
                last_players   = players
                last_rackets = rackets
                last_ball      = ball
                last_shot_type = shot_type
                last_velocity = velocity
                last_direction_change = direction_change
                
            # Visualization
            frame = visualizer.draw_players(frame, last_players)
            frame = visualizer.draw_ball(frame, last_ball)
            frame = visualizer.draw_rackets(frame, last_rackets)
            
            if last_shot_type is not None:
                frame = visualizer.draw_shot(frame, last_shot_type)
                
            ## out.write(frame)
            ## cv2.imshow('Padel Analytics', frame)
            
            display = dashboard.draw(
                frame, shot_results, last_velocity, last_direction_change, tracker.get_trail()
                )
            out.write(display)
            cv2.imshow('Padel Analytics', display)
            
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

        df = pd.DataFrame(shot_results) if shot_results else pd.DataFrame(columns=['frame','timestamp','shot_type', 'direction'])
        df.to_csv('../Outputs/result.csv', index=False)

        # Analytics
        shot_counts = Counter(s['shot_type'] for s in shot_results)

        analytics = {
            'total_shots'  : len(shot_results),
            'shot_counts'  : dict(shot_counts),
            'forehand_pct' : round(shot_counts.get('Forehand', 0) / max(len(shot_results), 1) * 100, 1),
            'backhand_pct' : round(shot_counts.get('Backhand', 0) / max(len(shot_results), 1) * 100, 1),
            'smash_pct'    : round(shot_counts.get('Smash',    0) / max(len(shot_results), 1) * 100, 1),
        }
        
        # Save analytics
        with open('../Outputs/analytics.json', 'w') as f:
            json.dump(analytics, f, indent=4)
        
        
        print(f'Processing Complete — {len(shot_results)} shots detected')
        print("\n--- SHOT ANALYTICS ---")
        for k, v in analytics.items():
            print(f"  {k}: {v}")

    
if __name__ == '__main__':
    main()
    
    
