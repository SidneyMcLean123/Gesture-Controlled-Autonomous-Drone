import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
from dronekit import connect, VehicleMode
import time

# SITL setup
#############################################
print("Connecting to vehicle...")
vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)
print("Connected!")
#############################################

def arm_and_takeoff(vehicle, target_altitude):
    print("Arming motors...")
    
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialize...")
        time.sleep(1)
    
    print("Switching to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    # wait until altitude reached
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.2f}")
        
        if alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        
        time.sleep(1)
        
def land_and_disarm(vehicle):
    print("Switching to LAND mode...")
    vehicle.mode = VehicleMode("LAND")
    
    while vehicle.mode.name != "LAND":
        print("Waiting for LAND mode...")
        time.sleep(1)
        
    print("Landing...")
    
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.2f}")
        
        if alt <= 0.1:
            print("Touchdown detected!")
            break
        
        time.sleep(1)
    
    print("Disarming...")
    vehicle.armed = False
    
    while vehicle.armed:
        print("Waiting for disarm...")
        time.sleep(1)

    print("Disarmed successfully.")
    
def hover_and_switch(vehicle, hover_time=10):
    print("Switching to GUIDED mode (hover)...")
    vehicle.mode = VehicleMode("GUIDED")
    
    while vehicle.mode.name != "GUIDED":
        print("Waiting for GUIDED mode...")
        time.sleep(1)

    print("Hovering...")

    start_time = time.time()
    
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Hover altitude: {alt:.2f}")

        # optional: time-based hover exit
        if time.time() - start_time > hover_time:
            print("Hover time complete")
            break

        time.sleep(1)

# decision layer setup
MIN_FRAMES_BEFORE_DECISION = 5
GESTURE_BUFFER_LEN = 15
GESTURE_BUFFER_CONFIDENCE_THRESHOLD = 0.6
state = "LANDED"
last_gesture = None
last_command = None
gesture_buffer = deque(maxlen=GESTURE_BUFFER_LEN)

# map gestures to intents
GESTURE_TO_INTENT = {
    "open_palm" : "HOVER",
    "closed_fist" : "LAND",
    "point_up" : "TAKEOFF"
}

# function to gesture from series of frames
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
    
# new command when intent serves as a state transition
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
    

# load model and encoder
model = tf.keras.models.load_model("gesture_model.keras", compile=False)
encoder = joblib.load("label_encoder.pkl")

# camera setup 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: couldn't open camera")
    exit()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

while True:
    command = None
    success, frame = cap.read()
    if not success:
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # feature extraction
        row = []
        for lm in hand_landmarks.landmark:
            row.append(lm.x)
            row.append(lm.y)

        X = np.array(row).reshape(1, -1)

        # prediction
        prediction = model.predict(X, verbose=0)
        class_id = np.argmax(prediction)

        label = encoder.inverse_transform([class_id])[0]
        
        # Decision Layer
        gesture_buffer.append(label)
        gesture = get_stable_gesture()
        
        if gesture and gesture != last_gesture:
            intent = GESTURE_TO_INTENT.get(gesture)
            state, command = update_state(state, intent)
            last_gesture = gesture

        if command and command != last_command:
            print(f"Executing: {command}")

            if command == "TAKEOFF":
                arm_and_takeoff(vehicle, 5)
    
            elif command == "LAND":
                land_and_disarm(vehicle)

            elif command == "HOVER":
                hover_and_switch(vehicle,10)
                
            last_command = command
        
        cv2.putText(frame, f"{gesture} | {state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):
        BGR_frame = cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('base_frame.jpg', BGR_frame)
        cv2.imwrite('landmark_frame.jpg', frame)

cap.release()
cv2.destroyAllWindows()