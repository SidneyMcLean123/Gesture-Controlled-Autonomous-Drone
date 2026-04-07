import cv2
import mediapipe as mp

# 0 = default camera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Can't receive frame")
        break

    # NOTE: opencv takes frame as BGR format instead of RGB like mediapipe
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)

    # for each hand identified in frame, draw the landmarks onto the frame
    if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Camera Test", frame)

    # key input handling 
    key = cv2.waitKey(1) & 0xFF

    # press q to quit
    if key == ord('q'):
        break

    # press spacebar to screenshot current frame (base + landmark)
    if key == ord(' '): 
        BRG_frame = cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('base_frame.jpg', BRG_frame)
        cv2.imwrite('landmark_frame.jpg', frame)

cap.release()     
cv2.destroyAllWindows()