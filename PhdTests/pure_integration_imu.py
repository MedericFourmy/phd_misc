#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import curves
from example_robot_data import loadTalos
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt; plt.ion()
from numpy.linalg import norm,inv,pinv,eig,svd

from wolf_preint import *


examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/tsid_gen/'
# examples_dir = ''
# file_name = 'sinY_nomove.cs'
# file_name = 'sinY_waist.cs'
# file_name = 'sinY_nowaist.cs'
file_name = 'sinY_nowaist_R3SO3.cs'

# INTTYPE = 'preint'
# INTTYPE = 'direct'
INTTYPE = 'directSE3'
# INTTYPE = 'pinocchio'

PATH = 'figs_'+INTTYPE+'/'

cs = ContactSequence()
cs.loadFromBinary(examples_dir + file_name)


q_traj   = cs.concatenateQtrajectories()
dq_traj  = cs.concatenateDQtrajectories()
ddq_traj = cs.concatenateDDQtrajectories()

min_ts = q_traj.min()
max_ts = q_traj.max()
print('traj dur (s): ', max_ts - min_ts)


dt = 1e-3  # discretization timespan
t_arr   = np.arange(min_ts, max_ts, dt)
q_arr   = np.array([q_traj(t) for t in t_arr])
dq_arr  = np.array([dq_traj(t) for t in t_arr])
ddq_arr = np.array([ddq_traj(t) for t in t_arr])

N = t_arr.shape[0]  # nb of discretized timestamps

# Load robot model
robot = loadTalos()
rmodel = robot.model
rdata = robot.data

# initialize 
oRb = pin.Quaternion(q_arr[0,3:7].reshape((4,1))).toRotationMatrix()

# init est and gtr lists
p_est_lst = []
oRb_est_lst = []
v_est_lst = []

p_gtr_lst = []
oRb_gtr_lst = []
v_gtr_lst = []

p_int = q_arr[0,0:3]
v_int = oRb @ dq_arr[0,0:3]
oRb_int = oRb.copy()
q_int = q_arr[0,:]

# Preintegration
p_ori = q_arr[0,0:3]
v_ori = oRb @ dq_arr[0,0:3]
oRb_ori = oRb.copy()
x_imu_ori = p_ori, v_ori, oRb_ori
DeltaIMU = [
    np.zeros(3),
    np.zeros(3),
    np.eye(3)
]
Deltat = 0

cur_nu_int = pin.Motion(dq_arr[0,:3], dq_arr[0,3:6])
o_M_cur = pin.SE3(quatarr_to_rot(q_arr[0,3:7]), q_arr[0,:3])

q_int = q_arr[0,:].copy()
dq_int = dq_arr[0,:].copy()


for i in range(0,N):

    # Store estimation
    p_est_lst.append(p_int.copy())
    oRb_est_lst.append(oRb_int.copy())
    v_est_lst.append(v_int.copy())

    p_gtr_lst.append(q_arr[i,0:3])
    o_q_b = q_arr[i,3:7]
    oRb = quatarr_to_rot(o_q_b)
    oRb_gtr_lst.append(oRb)
    v_gtr_lst.append(oRb @ dq_arr[i,0:3])

    # IMU measurements
    b_w, b_acc, b_proper_acc = imu_meas(dq_arr[i,:], ddq_arr[i,:], oRb)

    if INTTYPE == 'preint':
        ################
        # preintegration
        ################
        Deltat += dt
        deltak = compute_current_delta_IMU(b_proper_acc, b_w, dt)
        DeltaIMU = compose_delta_IMU(DeltaIMU, deltak, dt)
        p_int, v_int, oRb_int = state_plus_delta_IMU(x_imu_ori, DeltaIMU, Deltat)

    elif INTTYPE == 'direct':
        ########################
        # integrate one step IMU
        ########################
        p_int = p_int + v_int*dt + 0.5*oRb_int @ b_acc*dt**2
        v_int = v_int + oRb_int @ b_acc*dt
        oRb_int = oRb_int @ pin.exp(b_w*dt)
    
    elif INTTYPE == 'directSE3':
        cur_nu_int += pin.Motion(ddq_arr[i,:6] * dt)
        cur_M_next = pin.exp6(dq_arr[i,:6] * dt)  # integrate from ground truth conf vel
        # In fact we "know" the ground truth for angular vel:
        # cur_nu_int.angular = dq_arr[i,3:6]
        # cur_M_next = pin.exp6(cur_nu_int * dt)
        o_M_cur = o_M_cur * cur_M_next
        p_int = o_M_cur.translation
        oRb_int = o_M_cur.rotation
        v_int = oRb_int @ cur_nu_int.linear

    
    elif INTTYPE == 'pinocchio':
        ### INT IN SE3
        # q_int = pin.integrate(robot.model, q_int, dq_arr[i,:]*dt)
        q_int = pin.integrate(robot.model, q_int, dq_int*dt)
        dq_int += ddq_arr[i,:]*dt
        p_int = q_int[:3]
        oRb_int = quatarr_to_rot(q_int[3:7])
        v_int = oRb_int@dq_int[:3]

    # MCAPI integration scheme
    # dq_int += ddq_arr[i,:]*dt/2
    # q_int = pin.integrate(robot.model,q_int,dq_int*dt)
    # dq_int += ddq_arr[i,:]*dt/2
    # q_int = pin.integrate(robot.model,q_int,dq_int*dt)
    # dq_int += ddq_arr[i,:]*dt/2


# store in arrays what can be
p_est_arr = np.array(p_est_lst)
v_est_arr = np.array(v_est_lst)
o_est_arr = np.array([pin.log3(oRb_est) for oRb_est in oRb_est_lst])

p_gtr_arr = np.array(p_gtr_lst)
v_gtr_arr = np.array(v_gtr_lst)
o_gtr_arr = np.array([pin.log3(oRb_gtr) for oRb_gtr in oRb_gtr_lst])


# compute errors
p_err_arr = p_est_arr - p_gtr_arr
v_err_arr = v_est_arr - v_gtr_arr
oRb_err_arr = np.array([pin.log3(oRb_est @ oRb_gtr.T) for oRb_est, oRb_gtr in zip(oRb_est_lst, oRb_gtr_lst)])



print('Save figs in '+PATH)
#######
# PLOTS
#######
# errors
err_arr_lst = [p_err_arr, oRb_err_arr, v_err_arr]
fig_titles = ['P error', 'O error', 'V error']
for err_arr, fig_title in zip(err_arr_lst, fig_titles):
    fig = plt.figure(fig_title)
    for axis, (axis_name, c) in enumerate(zip('xyz', 'rgb')):
        plt.plot(t_arr, err_arr[:,axis], c, label='err_'+axis_name)
        plt.legend()
    fig.suptitle(fig_title)
    fig.savefig(PATH+fig_title+'.png')

# ground truth vs est
est_arr_lst = [p_est_arr, o_est_arr, v_est_arr]
gtr_arr_lst = [p_gtr_arr, o_gtr_arr, v_gtr_arr]
fig_titles = ['P est vs gtr', 'O est vs gtr', 'V est vs gtr']
for est_arr, gtr_arr, fig_title in zip(est_arr_lst, gtr_arr_lst, fig_titles):
    fig = plt.figure(fig_title)
    for axis, (axis_name, c) in enumerate(zip('xyz', 'rgb')):
        plt.subplot(3,1,axis+1)
        plt.plot(t_arr, est_arr[:,axis], c+':', label='est_'+axis_name)
        plt.plot(t_arr, gtr_arr[:,axis], c, label='gtr_'+axis_name)
        plt.legend()
    fig.suptitle(fig_title)
    fig.savefig(PATH+fig_title+'.png')

plt.show()
