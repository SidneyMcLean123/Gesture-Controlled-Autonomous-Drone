from dronekit import VehicleMode
import time

TARGET_ALTITUDE = 2    # in meters

# Arm and ascend to TARGET_ALTITUDE (where it hovers with guided)
def handle_takeoff(vehicle):
    if not vehicle.is_armable:
        print("Waiting for vehicle to become armable...")
        return
    
    if vehicle.mode != VehicleMode("GUIDED"):
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    if not vehicle.armed:
        print("Arming...")
        vehicle.armed = True
        while not vehicle.armed:
            time.sleep(0.5)
        print("Armed!")
        vehicle.simple_takeoff(TARGET_ALTITUDE)
        print(f"Taking off to {TARGET_ALTITUDE}m...")

# Hold position        
def handle_hover(vehicle):
    """Hold position — GUIDED mode with no new commands keeps the drone in place."""
    if vehicle.mode != VehicleMode("GUIDED"):
        vehicle.mode = VehicleMode("GUIDED")

# Landing sequence
def handle_land(vehicle):
    """Initiate landing sequence."""
    if vehicle.mode != VehicleMode("LAND"):
        print("Landing...")
        vehicle.mode = VehicleMode("LAND")

# Cut motors no matter the altitude
def handle_emergency(vehicle):
    """Cut motors immediately regardless of altitude."""
    print("EMERGENCY STOP")
    vehicle.mode = VehicleMode("STABILIZE")
    vehicle.armed = False
