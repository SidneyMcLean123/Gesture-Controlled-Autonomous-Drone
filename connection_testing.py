from dronekit import connect

vehicle = connect('udp:0.0.0.0:14552', wait_ready=True)

print("Connected!")
print("Mode:", vehicle.mode.name)
print("Armed:", vehicle.armed)

vehicle.close()