# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

N_SIMULATION = 10000             # number of time steps simulated
dt = 0.001                      # controller time step

mu = 0.3                            # friction coefficient
fMin = 0.0                          # minimum normal force
fMax = 500.0                       # maximum normal force

reference_config_q_name = 'standing'

# contact_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
contact_frame_names = ["FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE"]  # same thing as XX_FOOT but contained in pybullet
contact6d = False
useLocalFrame = True
contactNormal = np.array([0., 0., 1.])       # direction of the normal to the contact surface

priority_com = 0  # if set to 1, w_com has to be orders of magnitude above other weights
priority_foot = 1
priority_posture = 1
priority_forceRef = 1
priority_trunk = 1
priority_torque_bounds = 0
priority_joint_bounds = 0

w_com = 10.0                       # weight of center of mass task
w_foot = 10                        # weight of the foot motion task
w_posture = 1e-1                  # weight of joint posture task
w_forceRef = 1e-3                 # weight of force regularization task
w_trunk = 1.0                     # weight of trunk task
# w_torque_bounds = 1.0             # weight of the torque bounds
# w_joint_bounds = 0.0

kp_contact = 30.0                 # proportional gain of contact constraint
kp_foot = 10000.0                    # proportional gain of contact constraint
kp_com = 20.0                     # proportional gain of center of mass task               
kp_trunk = 500.0                  # proportional gain of trunk task

kd_com = 2.0 * np.sqrt(kp_com)

tau_max_scaling = 1.45            # scaling factor of torque bounds
v_max_scaling = 0.8

kp_posture = np.array(
    [ 10.,  5., 5.]*4                  
).T 


VIEWER_ON = True
PRINT_N = 500                   # print every PRINT_N time steps
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [1.5839792490005493,
 -1.4289098978042603,
 0.49806731939315796,
 0.5652844309806824,
 0.25840288400650024,
 0.3108690679073334,
 0.719056248664856]

SPHERE_RADIUS = 0.03
REF_SPHERE_RADIUS = 0.03
COM_SPHERE_COLOR  = (1, 0.8, 0, 0.5)
COM_REF_SPHERE_COLOR  = (1, 0, 0, 0.5)
F_SPHERE_COLOR  = (0, 1, 0, 0.5)
F_REF_SPHERE_COLOR  = (0, 0, 1, 0.5)

path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'
srdf = path + '/srdf/solo.srdf'    