import time
from dronekit import connect, VehicleMode

# dronekit commands state=LANDED and intent=TAKEOFF
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
        
# dronekit commands state=HOVERING and intent=HOVER
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
        
# dronekit commands state=HOVERING and intent=LAND        
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
