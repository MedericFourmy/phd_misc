import time
import vrpn

FREQ = 120
dt = 4/FREQ

#set according to Optitrack configuration
optitrack_rigid_body_name0="robot"
optitrack_server_address="192.168.101.93"

def callback0(userdata, data):
    print()
    print(data['time'])
    print(data['position'])
    print(data['quaternion'])
    print(userdata, " => ", data)

    
tracker0=vrpn.receiver.Tracker(optitrack_rigid_body_name0+"@"+optitrack_server_address)
tracker0.register_change_handler("position", callback0, "position")

while 1:
    tracker0.mainloop()
    # time.sleep(dt)
    # print('\n')
