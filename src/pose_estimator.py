# extract body keypoints using mediapipe pose

import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 1,
            smooth_landmarks = True,
            enable_segmentation = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        
        
    def estimate(self, frame):
        if frame is None:
            return {}
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(rgb)
        
        landmarks = {}
        
        if results and results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = (lm.x, lm.y)
        
        
        return landmarks
    
