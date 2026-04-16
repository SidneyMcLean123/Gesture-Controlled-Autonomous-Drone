import cv2
import mediapipe as mp
import csv

# 0 = default camera
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# current_label - switch using number keys
current_label = "closed_fist"

# csv setup for appending data capture
with open('raw_gesture_data_collection.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)

        # draw landmarks on frame
        if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Camera", frame)

        # key input handling 
        key = cv2.waitKey(1) & 0xFF

        # press q to quit
        if key == ord('q'):
            break

        # press spacebar to capture landmark 
        if key == ord(' ') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include current_label
            row.append(f"{current_label}")

            writer.writerow(row)
            print(f"{current_label} captured!")
            
        # press 1 to change current_label to closed_fist
        if key == ord('1'):
            current_label = "closed_fist"
            print(f"current_label changed to: {current_label}")
            
        # press 2 to change current_label to closed_fist
        if key == ord('2'):
            current_label = "open_palm"
            print(f"current_label changed to: {current_label}")
        
        # press 3 to change current_label to closed_fist
        if key == ord('3'):
            current_label = "point_up"
            print(f"current_label changed to: {current_label}")
            
        # press 4 to change current_label to closed_fist
        if key == ord('4'):
            current_label = "point_down"
            print(f"current_label changed to: {current_label}")
            
        # press 5 to change current_label to closed_fist
        if key == ord('5'):
            current_label = "point_left"
            print(f"current_label changed to: {current_label}")
        
        # press 6 to change current_label to closed_fist
        if key == ord('6'):
            current_label = "point_right"
            print(f"current_label changed to: {current_label}")
            
        # press 7 to change current_label to closed_fist
        if key == ord('7'):
            current_label = "peace_sign"
            print(f"current_label changed to: {current_label}")

cap.release()     
cv2.destroyAllWindows()