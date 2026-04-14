from collections import deque
from typing import Optional

# Decision Configuration 
GESTURE_BUFFER_LEN = 15     # frames to smooth over (smoothing quality)
MIN_FRAMES = 5              # frames needed before decision 
CONFIDENCE = 0.6

GESTURE_TO_INTENT = {
    "open_palm" : "HOVER",
    "closed_fist" : "LAND",
    "point_up" : "TAKEOFF"
}

class Decision:
    def __init__(self):
        self.window = deque(maxlen=GESTURE_BUFFER_LEN)
        self.last_stable = None
    
    def update(self, label: str) -> Optional[str]:
        self.window.append(label)
        
        if len(self.window) < MIN_FRAMES:
            return None
        
        most_common = max(set(self.window), key=self.window.count)
        
        if self.window.count(most_common) / len(self.window) >= CONFIDENCE and most_common != self.last_stable:
            self.last_stable = most_common
            return GESTURE_TO_INTENT.get(most_common)
        
        return None
