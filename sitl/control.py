import time
from dronekit import connect, VehicleMode

# dronekit commands, state=LANDED and intent=TAKEOFF
def handle_takeoff(vehicle, target_altitude):
    global state, takeoff_started
    
    alt = vehicle.location.global_relative_frame.alt
    
    # resloved 
    if not vehicle.is_armable:
        print("Waiting for vehicle to become armable")
        return
    
    if vehicle.is_armable:
        print("Vehicle is armable")
        vehicle.mode = VehicleMode("GUIDED")
        print("Vehicle Mode: %s" % VehicleMode)
    
    if not vehicle.armed:
        print("Arming...")
        vehicle.armed = True
        return
    
    if not takeoff_started:
        print("Taking off...")
        vehicle.simple_takeoff(target_altitude)
        takeoff_started = True
        return
    
    print(f"Altitude: {alt:.2f}")
    
    if alt >= target_altitude:
        print("Reached altitude")
        print("Switching to HOVER")
        state = State.HOVER
        takeoff_started = False
        
# dronekit commands, state=HOVERING and intent=HOVER
def handle_hover(vehicle, hover_time=10):
    alt = vehicle.location.global_relative_frame.alt
    print(f"Altitude: {alt:.2f}")
        
# dronekit commands, state=HOVERING and intent=LAND        
def handle_land(vehicle):
    global state
    
    if vehicle.mode.name != "LAND":
        print("Switching to LAND...")
        vehicle.mode = VehicleMode("LAND")
        return

    alt = vehicle.location.global_relative_frame.alt
    print(f"Landing... Alt: {alt:.2f}")

    if alt <= 0.1:
        print("Touchdown")
        print("Disarming")
        vehicle.armed = False
        state = State.IDLE

# dronekit commands, EMERGENCY STOP
def handle_emergency(vehicle):
    global state
    
    print("EMERGENCY STOP")
    
    vehicle.mode = VehicleMode("LAND")
    vehicle.armed = False
    state = State.IDLE