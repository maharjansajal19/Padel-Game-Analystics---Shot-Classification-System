import cv2
import numpy as np
from collections import Counter

class Dashboard:
    def __init__(self, width=400, height=600):
        self.width = width
        self.height = height
        
    def draw(self, frame, shot_results, velocity, direction_change, ball_trail):
        
        # Create dark panel on right side
        h, w = frame.shape[:2]
        panel = np.zeros((h, self.width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30) # dark background
        
        # Title
        cv2.putText(panel, 'PADEL ANALYTICS', (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.line(panel, (10, 45), (self.width-10, 45), (60, 60, 60), 1)
        
        # Shot counts
        counts = Counter(s['shot_type'] for s in shot_results)
        cv2.putText(panel, 'SHOT COUNTS', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        colors = {'Forehand': (0, 255, 0), 'Backhand': (255, 165, 0),
                  'Smash': (0, 0, 255), 'Bounce': (255, 255, 0)}
        
        y = 110
        for shot, color in colors.items():
            count = counts.get(shot, 0)
            total = max(len(shot_results), 1)
            pct = int(count / total * 100)
            
            cv2.putText(panel, f'{shot}: {count} ({pct}%)', (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            
            # Progress bar
            bar_w = int((self.width - 20) * count / max(total, 1))
            cv2.rectangle(panel, (10, y+5), (self.width-10, y+15), (60, 60, 60), -1)
            cv2.rectangle(panel, (10, y+5), (10+bar_w, y+15), color, -1)
            y += 40
            
        cv2.line(panel, (10, y), (self.width-10, y), (60, 60, 60), 1)
        y += 20
        
        # Live stats
        cv2.putText(panel, 'LIVE STATS', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 30
        cv2.putText(panel, f'Velocity    : {round(velocity, 1)} px/f', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255, 255, 255), 1)
        y += 25
        cv2.putText(panel, f'Dir Change  : {round(direction_change, 1)} deg', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(panel, f'Total Shots : {len(shot_results)}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ball trail on mini court
        y += 40
        cv2.putText(panel, 'BALL TRAIL', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 10
        court_x, court_y = 10, y
        court_w, court_h = self.width-20, 150
        cv2. rectangle(panel, (court_x, court_y),
                       (court_x + court_w, court_y + court_h), (60, 60, 60), -1)
        cv2.rectangle(panel, (court_x, court_y),
                      (court_x + court_w, court_y + court_h), (100, 100, 100), 1)
        
        # Draw trail dots
        if ball_trail:
            for i, (bx, by) in enumerate(ball_trail):
                # Normalize to mini court
                nx = int(bx / 1920 * court_w) + court_x
                ny = int(by / 1080 * court_h) + court_y
                alpha = int(255 * i / len(ball_trail))
                cv2.circle(panel, (nx, ny), 2, (0, alpha, 255), -1)
                
        # Combine frame + panel
        combined = np.hstack([frame, panel])
        return combined
        