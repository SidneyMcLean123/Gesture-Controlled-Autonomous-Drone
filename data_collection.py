import cv2
import mediapipe as mp
import csv

# 0 = default camera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# csv setup for appending data capture
with open('raw_gesture_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Can't receive frame")
            break

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)

        # draw landmarks in real time for webcam
        if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Camera Test", frame)

        # key input handling 
        key = cv2.waitKey(1) & 0xFF

        # press q to quit
        if key == ord('q'):
            break

        # press 1 for closed_fist landmark capture
        if key == ord('1') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("closed_fist")

            writer.writerow(row)
            print(f"closed_fist captured!")
        # press 2 for open palm landmark capture
        if key == ord('2') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("open_palm")

            writer.writerow(row)
            print(f"open_palm captured!")
        # press 3 for point up landmark capture
        if key == ord('3') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("point_up")

            writer.writerow(row)
            print(f"point_up captured!")
        # press 4 for point down landmark capture
        if key == ord('4') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("point_down")

            writer.writerow(row)
            print(f"point_down captured!")
        # press 5 for point right landmark capture
        if key == ord('5') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("point_right")

            writer.writerow(row)
            print(f"point_right captured!")
        # press 6 for point left landmark caputre
        if key == ord('6') and result.multi_hand_landmarks:
            # grab one hand if multiple
            hand_landmarks = result.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # include label
            row.append("point_left")

            writer.writerow(row)
            print(f"point_left captured!")


cap.release()     
cv2.destroyAllWindows()