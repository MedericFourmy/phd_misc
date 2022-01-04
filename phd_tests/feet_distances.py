import pinocchio as pin
from example_robot_data import load
import numpy as np

robot = load('solo12')

q = np.array(6*[0]+[1]+12*[0])

LEGS = ['FL', 'FR', 'HL', 'HR']
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

combinations = [
    (0,1),
    (1,2),
    (2,3),
    (0,3),
    # (0,2),
    # (1,3),
]

for i, comb in enumerate(combinations):
    p0 = robot.framePlacement(q, cids[comb[0]], update_kinematics=True).translation
    p1 = robot.framePlacement(q, cids[comb[1]], update_kinematics=True).translation

    dist = np.linalg.norm(p0 - p1)
    print(LEGS[comb[0]], LEGS[comb[1]], dist)