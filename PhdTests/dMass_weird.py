import numpy as np
import pinocchio as pin
from pinocchio.utils import rand
from example_robot_data import loadANYmal

robot = loadANYmal()
rm = robot.model
rd = robot.data
q = robot.q0
qv = rand(robot.nv)

C = pin.computeCoriolisMatrix(rm, rd, q, qv)
dM_dt = pin.dccrba(rm, rd, q, qv)

# verify dM/dt = C + C.T
C_sum = C + C.T

# Why is dM_dt only 6 x nqv and not nqv x nqv?
print(dM_dt.shape)

# not zero
print(np.linalg.norm(C_sum[:6,:] - dM_dt))