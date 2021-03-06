# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

N_SIMULATION = 10000             # number of time steps simulated
dt = 1e-3                      # controller time step
# N_SIMULATION = 1000             # number of time steps simulated
# dt = 1e-2                      # controller time step

lxp = 0.1                           # foot length in positive x direction
lxn = 0.1                           # foot length in negative x direction
lyp = 0.065                         # foot length in positive y direction
lyn = 0.065                         # foot length in negative y direction
lz = 0.107                          # foot sole height with respect to ankle joint
mu = 0.3                            # friction coefficient
fMin = 0.0                          # minimum normal force
fMax = 2000.0                       # maximum normal force

reference_config_q_name = 'half_sitting'

contact_frame_names = ['leg_left_6_joint', 'leg_right_6_joint']
contact6d = True
contactNormal = np.array([0., 0., 1.])       # direction of the normal to the contact surface

priority_com = 0  # if set to 1, w_com has to be orders of magnitude above other weights
priority_foot = 1
priority_posture = 1
priority_forceRef = 1
priority_trunk = 1
priority_torque_bounds = 0
priority_joint_bounds = 0

w_com = 1.0                       # weight of center of mass task
w_foot = 1                        # weight of the foot motion task
w_posture = 1e-1                  # weight of joint posture task
w_forceRef = 1e-3                 # weight of force regularization task
w_trunk = 1.0                     # weight of trunk task
w_torque_bounds = 1.0             # weight of the torque bounds
w_joint_bounds = 0.0

kp_contact = 30.0                 # proportional gain of contact constraint
kp_foot = 100.0                   # proportional gain of contact constraint
kp_com = 20.0                     # proportional gain of center of mass task               
kp_trunk = 500.0                  # proportional gain of trunk task

kd_com = 2.0 * np.sqrt(kp_com)
# kd_com = 0   # useful if we don't have a desired com vel

tau_max_scaling = 1.45            # scaling factor of torque bounds
v_max_scaling = 0.8

kp_posture = np.array(                                    # proportional gain of joint posture task
    [ 10. ,  5.  , 5. , 1. ,  1. ,  10.,                  # lleg  #low gain on axis along y and knee
    10. ,  5.  , 5. , 1. ,  1. ,  10.,                    # rleg
    500. , 500.  ,                                        # chest
    50.,   10.  , 10.,  10.,    10. ,  10. , 10. ,  10. , # larm
    50.,   10.  , 10., 10.,    10. ,  10. ,  10. ,  10. , # rarm
    100.,  100.]                                          # head
).T 

VIEWER_ON = False
PRINT_N = 500                   # print every PRINT_N time steps
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [4.0, -0.2, 0.4, 0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333]


SPHERE_RADIUS = 0.03
REF_SPHERE_RADIUS = 0.03
COM_SPHERE_COLOR  = (1, 0.8, 0, 0.5)
COM_REF_SPHERE_COLOR  = (1, 0, 0, 0.5)
F_SPHERE_COLOR  = (0, 1, 0, 0.5)
F_REF_SPHERE_COLOR  = (0, 0, 1, 0.5)

path = '/opt/openrobots/share/example-robot-data/robots/talos_data'
urdf = path + '/robots/talos_reduced.urdf'
srdf = path + '/srdf/talos.srdf'