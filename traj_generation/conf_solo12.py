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

contact_frame_names = ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT']
contact6d = False
contactNormal = np.array([0., 0., 1.])       # direction of the normal to the contact surface

priority_com = 0  # if set to 1, w_com has to be orders of magnitude above other weights
priority_foot = 1
priority_posture = 1
priority_forceRef = 1
priority_waist = 1
priority_torque_bounds = 0
priority_joint_bounds = 0

w_com = 10.0                       # weight of center of mass task
w_foot = 10                        # weight of the foot motion task
w_posture = 1e-1                  # weight of joint posture task
w_forceRef = 1e-3                 # weight of force regularization task
w_waist = 1.0                     # weight of waist task
w_torque_bounds = 1.0             # weight of the torque bounds
w_joint_bounds = 0.0

kp_contact = 30.0                 # proportional gain of contact constraint
kp_foot = 10000.0                    # proportional gain of contact constraint
kp_com = 20.0                     # proportional gain of center of mass task               
kp_waist = 500.0                  # proportional gain of waist task

kd_com = 2.0 * np.sqrt(kp_com)

tau_max_scaling = 1.45            # scaling factor of torque bounds
v_max_scaling = 0.8

kp_posture = np.array(
    [ 10.,  5., 5.]*4                  
).T 


VIEWER_ON = False
PRINT_N = 500                   # print every PRINT_N time steps
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [0.016945913434028625, -2.2850289344787598, 0.3789125978946686,
                    0.6462744474411011, 0.023162445053458214, -0.021665913984179497, 0.7624456882476807]

SPHERE_RADIUS = 0.03
REF_SPHERE_RADIUS = 0.03
COM_SPHERE_COLOR  = (1, 0.8, 0, 0.5)
COM_REF_SPHERE_COLOR  = (1, 0, 0, 0.5)
F_SPHERE_COLOR  = (0, 1, 0, 0.5)
F_REF_SPHERE_COLOR  = (0, 0, 1, 0.5)

path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'
srdf = path + '/srdf/solo.srdf'    