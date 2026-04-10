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

# from decision import MIN_FRAMES_BEFORE_DECISION, GESTURE_BUFFER_LEN, GESTURE_BUFFER_CONFIDENCE_THRESHOLD, gesture_buffer, get_stable_gesture, update_state
# from control import arm_and_takeoff, land_and_disarm, hover_and_switch

# SITL setup
#############################################
print("Connecting to vehicle...")
vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)
print("Connected!")
#############################################

# decision layer setup
state = "LANDED"
last_gesture = None
last_command = None

# map gestures to intents
GESTURE_TO_INTENT = {
    "open_palm" : "HOVER",
    "closed_fist" : "LAND",
    "point_up" : "TAKEOFF"
}
    
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
        
        # Decision/Control Layer
        decision.gesture_buffer.append(label)
        gesture = decision.get_stable_gesture()
        
        if gesture and gesture != last_gesture:
            intent = GESTURE_TO_INTENT.get(gesture)
            state, command = decision.update_state(state, intent)
            last_gesture = gesture

        if command and command != last_command:
            print(f"Executing: {command}")

            if command == "TAKEOFF":
                control.arm_and_takeoff(vehicle, 5)
    
            elif command == "LAND":
                control.land_and_disarm(vehicle)

            elif command == "HOVER":
                control.hover_and_switch(vehicle,10)
                
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