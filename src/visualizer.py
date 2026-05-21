# Draws overlays

import cv2

class Visualizer:
    
    def draw_players(self, frame, players):
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)
            
            cv2.putText(frame, 'Player',
                        (x1, y1 -8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 0),
                        1)
            
        return frame
    
    def draw_rackets(self, frame, rackets):
        for racket in rackets:
            x1, y1, x2, y2 = racket['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)  # orange
            cv2.putText(frame, 'Racket', (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)
        return frame
    
    def draw_ball(self, frame, ball):
        if ball is not None:
            x1, y1, x2, y2 = ball['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw circle instead of rectangle for ball
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
            cv2.putText(frame, 'Ball',
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1)
            
        return frame
    
    def draw_shot(self, frame, shot_type):
        colors = {
            'Smash': (0, 0, 255),
            'Forehand': (0, 255, 0),
            'Backhand': (255, 165, 0)
        }
        color = colors.get(shot_type, (255, 255, 255))
        
        cv2.putText(frame,
                    f'Shot:{shot_type}',
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    3)
        
        return frame