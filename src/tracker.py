# tracks the ball position over time

from collections import deque
import numpy as np

class BallTracker:
    def __init__(self, max_length = 30):
        self.positions = deque(maxlen=max_length)
        self.missing_frames = 0
        self.MAX_MISSING = 5 # reset trail after 5 missed frames
        
    def update(self, ball):
        if ball is not None:
            x1, y1, x2, y2 = ball['bbox']
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            self.positions.append((cx, cy))
            self.missing_frames = 0
        
        else:
            self.missing_frames += 1
            # Clear trail if ball missing too long (jumped to new position)
            if self.missing_frames > self.MAX_MISSING:
                self.positions.clear()
            
    def get_velocity(self):
        if len(self.positions) < 2:
            return 0
        
        p1 = np.array(self.positions[-2])
        p2 = np.array(self.positions[-1])
        
        velocity = float(np.linalg.norm(p2 - p1))
        
        return velocity
    
    def get_direction_change(self):
        if len(self.positions) < 3:
            return 0
        
        p1 = np.array(self.positions[0])
        p2 = np.array(self.positions[len(self.positions)//2])
        p3 = np.array(self.positions[-1])
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return 0
        
        cosine = np.dot(v1, v2)/ (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        
        angle = np.arccos(np.clip(cosine, -1, 1))
        
        return float(np.degrees(angle))
    
    # Bounce detection
    def detect_bounce(self):
        if len(self.positions) < 3:
            return False
        
        p1 = self.positions[-3]
        p2 = self.positions[-2] # potential bounce point
        p3 = self.positions[-1]
        
        # Ball goes down then up = bounce (y increases then decreases)
        going_down = p2[1] > p1[1]
        going_up = p3[1] < p2[1]
        
        return going_down and going_up
    
    def get_trail(self):
        """Returns positions list for visualizing ball trail"""
        return list(self.positions)