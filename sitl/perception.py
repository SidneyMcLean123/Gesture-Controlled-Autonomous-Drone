import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf

class Perception:
    def __init__(self):
        self.model = tf.keras.models.load_model("gesture_model.keras", compile=False)
        self.encoder = joblib.load("label_encoder.pkl")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: couldn't open camera")
    
    # Read one frame and return (frame, label)
    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        label = None
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y])

            X = np.array(row).reshape(1, -1)
            prediction = self.model.predict(X, verbose=0)
            class_id = np.argmax(prediction)
            label = self.encoder.inverse_transform([class_id])[0]

        return frame, label
    
    # Cleanup for cv2
    def release(self):
        self.cap.release()