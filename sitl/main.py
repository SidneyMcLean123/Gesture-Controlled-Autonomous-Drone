import cv2
import threading
import time
from dronekit import connect, VehicleMode
from perception import Perception
from decision import Decision
from state_machine import StateMachine, State
import control

# SITL setup
#############################################
print("Connecting to vehicle...")
vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)
print("Connected!")
#############################################

# Layer Setups 
perception = Perception()
sm = StateMachine()
decision = Decision()

# Control Loop definition
def control_loop(vehicle, sm):
    while True:
        current = sm.get_state()

        if current == State.TAKEOFF:
            control.handle_takeoff(vehicle)

            # Auto-transition to HOVER once altitude is reached
            if vehicle.location.global_relative_frame.alt >= control.TARGET_ALTITUDE:
                sm.transition("HOVER")

        elif current == State.HOVER:
            control.handle_hover(vehicle)

        elif current == State.LAND:
            control.handle_land(vehicle)

            # Auto-transition to IDLE once landed
            if vehicle.location.global_relative_frame.alt <= 0.3:
                sm.transition("IDLE")  # requires adding LAND->IDLE in transitions

        elif current == State.EMERGENCY_STOP:
            control.handle_emergency(vehicle)

        time.sleep(0.1)

# Control thread to start Control Loop
threading.Thread(target=control_loop, args=(vehicle, sm), daemon=True).start()

# Perception and Decision Loop (Main Loop)
try:
    while True:
        frame, label = perception.get_frame()
        if frame is None:
            break
        
        if label: 
            intent = decision.update(label)
            if intent:
                sm.transition(intent)
            cv2.putText(frame, f"{label} | {sm.get_state().name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
    vehicle.close()
    print("Vehicle Connection Closed")



