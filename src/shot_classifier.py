# classifies the shot

import numpy as np

class ShotClassifier:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0
    
    def __init__(self):
        pass
    
    def classify(self, landmarks, velocity, direction_change):
        
        ## if len(landmarks) == 0:
        ##     return None
        
        if len(landmarks) > 0:
            try:
                right_wrist = landmarks.get(self.RIGHT_WRIST)
                left_wrist = landmarks.get(self.LEFT_WRIST)
                right_shoulder = landmarks.get(self.RIGHT_SHOULDER)
                left_shoulder = landmarks.get(self.LEFT_SHOULDER)
                nose = landmarks.get(self.NOSE)
                
                if not all([right_wrist, left_wrist, right_shoulder, left_shoulder, nose]):
                    return None
                
                rw_x, rw_y = right_wrist
                lw_x, lw_y = left_wrist
                rs_x, rs_y = right_shoulder
                ls_x, ls_y = left_shoulder
                _, no_y = nose
                
                # Use Whichever wrist is higher (lower y = higher on screen)
                active_wrist_y = min(rw_y, lw_y)
                shoulder_mid_y = (rs_y + ls_y) / 2
                shoulder_mid_x = (rs_x + ls_x) / 2
                
                # Smash detection
                if active_wrist_y < no_y and velocity > 5:
                    return 'Smash'
                
                # Forehand detection
                if rw_x > shoulder_mid_x and direction_change > 10:
                    return 'Forehand'
                
                # Backhand detection
                if rw_x < shoulder_mid_x and direction_change > 10:
                    return 'Backhand'
                
            except Exception as e:
                print(f"[ShotClassifer] Error: {e}")
                return None
        
        # Fallback: ball motion only (works when landmarks empty)
        if velocity > 20 and direction_change < 15:
            return 'Smash'
        if direction_change > 25:
            return 'Forehand'
        if direction_change > 10:
            return 'Backhand'
        
        return None
    
    # Shot direction
    def get_shot_direction(self, positions):
        if len(positions) < 2:
            return None
        
        x1 = positions[0][0]
        x2 = positions[-1][0]
        
        if x2 > x1:
            return 'Right'
        elif x2 < x1:
            return 'Left'
        return 'Straight'