import time
import numpy as np
import pinocchio as pin
from example_robot_data import load

robot = load('solo12')

nqa = np.deg2rad([0.002])  # actual resolution of solo encoders
nqa = np.deg2rad([30])     # just to make it visible

q0 = robot.model.referenceConfigurations['standing']
q0[7:] = np.zeros(12)
q1 = q0.copy()
q1[7:] = q1[7:] + nqa*np.ones(12)

LEGS = ['FL', 'FR', 'HL', 'HR']
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

p0 = robot.framePlacement(q0, cids[0], update_kinematics=True).translation
p1 = robot.framePlacement(q1, cids[0], update_kinematics=True).translation


robot.initViewer(loadModel=True)

robot.display(q0)
time.sleep(5)
robot.display(q1)



print(np.linalg.norm(p0-p1))

