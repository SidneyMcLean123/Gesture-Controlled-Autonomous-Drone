from collections import deque
from enum import Enum
import threading

# Decision Configuration 
MIN_FRAMES_BEFORE_DECISION = 5
GESTURE_BUFFER_LEN = 15
GESTURE_BUFFER_CONFIDENCE_THRESHOLD = 0.6

# Buffer
gesture_buffer = deque(maxlen=GESTURE_BUFFER_LEN)

def add_gesture(gesture):
    gesture_buffer.append(gesture)

# get most popular gesture from buffer
def get_gesture():
    if len(gesture_buffer) < MIN_FRAMES_BEFORE_DECISION:
        return None
    # count number of each gesture
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    # get most common gesture
    gesture_candidate = max(counts, key=counts.get)
    # return only if gesture has majority confidence
    if counts[gesture_candidate] < len(gesture_buffer) * GESTURE_BUFFER_CONFIDENCE_THRESHOLD:
        return None
    
    return gesture_candidate
