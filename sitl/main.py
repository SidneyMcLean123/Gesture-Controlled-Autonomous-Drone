import cv2
from dronekit import connect, VehicleMode
from perception import Perception

# # SITL setup
# #############################################
# print("Connecting to vehicle...")
# vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)
# print("Connected!")
# #############################################

# Perception object
perception = Perception()
# last_gesture = None

try:
    while True:
        frame, label = perception.get_frame()
        if frame is None:
            break
        
        if label: 
        # annotate the frame with label
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Camera Test", frame)
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    # cleanup - Perception and release communication port
    perception.release()
    cv2.destroyAllWindows()
    # vehicle.close()
    # print("Vehicle Connection Closed")



