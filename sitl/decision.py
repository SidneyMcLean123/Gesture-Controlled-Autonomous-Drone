from collections import deque

MIN_FRAMES_BEFORE_DECISION = 5
GESTURE_BUFFER_LEN = 15
GESTURE_BUFFER_CONFIDENCE_THRESHOLD = 0.6

gesture_buffer = deque(maxlen=GESTURE_BUFFER_LEN)

# get most popular gesture from buffer
# gesture returned satisfies criteria:
#           - gesture is >= 60% of buffer
def get_stable_gesture():
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

# update state and report command
def update_state(state, intent):
    command = None
    
    if state == "LANDED":
        if intent == "TAKEOFF":
            command = "TAKEOFF"
            state = "TAKING_OFF"
    
    elif state == "TAKING_OFF":
        # immediate transition, only see when invalid gesture is presented for following state below
        state = "HOVERING"
            
    elif state == "HOVERING":
        if intent == "HOVER":
            command = "HOVER"
        elif intent == "LAND":
            command = "LAND"
            state = "LANDING"

    elif state == "LANDING":
        # immediate transition, only see when invalid gesture is presented for following state below
        state = "LANDED"
    
    return state, command