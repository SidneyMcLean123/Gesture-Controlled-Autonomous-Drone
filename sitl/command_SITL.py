import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
from dronekit import connect, VehicleMode
import time
import control
import decision
import threading
from enum import Enum

# SITL setup
#############################################
print("Connecting to vehicle...")
vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)
print("Connected!")
#############################################

# State Machine setup
class State(Enum):
    IDLE = 0
    TAKEOFF = 1
    HOVER = 2
    LAND = 3
    EMERGENCY_STOP = 4

state = State.IDLE
state_lock = threading.Lock()

def set_state(new_state):
    global state
    with state_lock:
        if state != new_state:
            print(f"STATE CHANGE: {state.name} → {new_state.name}")
            state = new_state

def get_state():
    with state_lock:
        return state

# Control Layer setup
takeoff_started = False
target_altitude = 10

def control_loop(vehicle):
    global state
    
    while True:
        if state == State.TAKEOFF:
            control.handle_takeoff(vehicle)

        elif state == State.HOVER:
            control.handle_hover(vehicle)

        elif state == State.LAND:
            control.handle_land(vehicle)

        elif state == State.EMERGENCY_STOP:
            control.handle_emergency(vehicle)

        time.sleep(0.1)  # important: don't max CPU

threading.Thread(target=control_loop, args=(vehicle,), daemon=True).start()

# perception layer setup
last_gesture = None
    
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
        decision.add_gesture(label)
        gesture = decision.get_gesture()
        
        if gesture and gesture != last_gesture:
            intent = decision.GESTURE_TO_STATE.get(gesture)

            if intent == "TAKEOFF":
                decision.update_state(State.TAKEOFF)
            elif intent == "HOVER":
                decision.update_state(State.HOVER)
            elif intent == "LAND":
                decision.update_state(State.LAND)

            last_gesture = gesture
        
        # display
        cv2.putText(frame, f"{gesture} | {state.name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()