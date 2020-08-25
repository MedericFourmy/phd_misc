#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import pinocchio as pin
from experiment_naming import dirname_from_params


# Select plots to activate
RUN_SIMULATION =        True
ERROR_BASE =            True    
ERROR_COM_AM_BIAS =     True
ANGULAR_MOMENTUM_BIAS = True
ABSOLUTE_BASE =         True
ABSOLUTE_COM_AM =       True
# WRENCH_PLOTS =          False
PLOT_SHOW =             True
# Saving figures
SAVE_FIGURES = True
FIG_DIR_PATH = 'figs/oneshot/'
SUB_DIR = dirname_from_params('/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/mcapi_povcdl_estimation.yaml')
PATH = FIG_DIR_PATH + SUB_DIR + "/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

if RUN_SIMULATION:
    t1 = time.time()
    subprocess.run('/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_povcdl_estimation', stdout=subprocess.DEVNULL)
    print('time: ', time.time()-t1)

df_gtr = pd.read_csv('gtr.csv')
df_est = pd.read_csv('est.csv')
df_kfs = pd.read_csv('kfs.csv')
df_wre = pd.read_csv('wre.csv')
# df_wre_tsid = pd.read_csv('wre_tsid.csv')

# COMPUTE ERROR ARRAYS
errpx = df_est['px'] - df_gtr['px']
errpy = df_est['py'] - df_gtr['py']
errpz = df_est['pz'] - df_gtr['pz']
# Extract quaternions to compute orientation difference (xyzw reversed in constructor -> normal)
quat_gtr_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_gtr['qx'], df_gtr['qy'], df_gtr['qz'], df_gtr['qw'])]
quat_est_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_est['qx'], df_est['qy'], df_est['qz'], df_est['qw'])]
#  q_est - q_gtr =def log3(q_gtr.inv * q_est)
quat_err_arr = np.array([pin.log3((q_gtr.inverse()*q_est).toRotationMatrix()) for q_gtr, q_est in zip(quat_gtr_lst, quat_est_lst)])
quat_err_mag = np.apply_along_axis(np.linalg.norm, 1, quat_err_arr)
errvx = df_est['vx'] - df_gtr['vx']
errvy = df_est['vy'] - df_gtr['vy']
errvz = df_est['vz'] - df_gtr['vz']
errcx = df_est['cx'] - df_gtr['cx']
errcy = df_est['cy'] - df_gtr['cy']
errcz = df_est['cz'] - df_gtr['cz']
errcdx = df_est['cdx'] - df_gtr['cdx']
errcdy = df_est['cdy'] - df_gtr['cdy']
errcdz = df_est['cdz'] - df_gtr['cdz']
errLx = df_est['Lx'] - df_gtr['Lx']
errLy = df_est['Ly'] - df_gtr['Ly']
errLz = df_est['Lz'] - df_gtr['Lz']

def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2))

print('RMSE errors')
print('px err:  ', rmse(errpx))
print('py err:  ', rmse(errpy))
print('pz err:  ', rmse(errpz))
print('ox err:  ', rmse(quat_err_arr[:,0]))
print('oy err:  ', rmse(quat_err_arr[:,1]))
print('oz err:  ', rmse(quat_err_arr[:,2]))
print('vx err:  ', rmse(errvx))
print('vy err:  ', rmse(errvy))
print('vz err:  ', rmse(errvz))
print('cx err:  ', rmse(errcx))
print('cy err:  ', rmse(errcy))
print('cz err:  ', rmse(errcz))
print('cdx err: ', rmse(errcdx))
print('cdy err: ', rmse(errcdy))
print('cdz err: ', rmse(errcdz))
print('Lx err:  ', rmse(errLx))
print('Ly err:  ', rmse(errLy))
print('Lz err:  ', rmse(errLz))



#############
# Error plots
#############
if ERROR_BASE:
    # base position
    fig = plt.figure('Base position error')
    plt.title('Base position error')
    plt.plot(df_est['t'], errpx, 'r', label='err_px')
    plt.plot(df_est['t'], errpy, 'g', label='err_py')
    plt.plot(df_est['t'], errpz, 'b', label='err_pz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'base_pos_err.png')

    # base orientation
    fig = plt.figure('Orientation error')
    plt.title('Orientation error')
    plt.plot(df_est['t'], quat_err_arr[:,0], 'r', label='err_orient_x')
    plt.plot(df_est['t'], quat_err_arr[:,1], 'g', label='err_orient_y')
    plt.plot(df_est['t'], quat_err_arr[:,2], 'b', label='err_orient_z')
    plt.plot(df_est['t'], quat_err_mag, 'y', label='err_orient_mag')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'base_orientation_err.png')

    # base vel
    fig = plt.figure('Base velocity error')
    plt.title('Base velocity error')
    plt.plot(df_est['t'], errvx, 'r', label='err_vx')
    plt.plot(df_est['t'], errvy, 'g', label='err_vy')
    plt.plot(df_est['t'], errvz, 'b', label='err_vz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'base_vel_err.png')

if ERROR_COM_AM_BIAS:
    # CoM position
    fig = plt.figure('COM position error')
    plt.title('COM position error')
    plt.plot(df_est['t'], errcx, 'r', label='err_cx')
    plt.plot(df_est['t'], errcy, 'g', label='err_cy')
    plt.plot(df_est['t'], errcz, 'b', label='err_cz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_pos_err.png')

    # CoM vel
    fig = plt.figure('COM velocity error')
    plt.title('COM velocity error')
    plt.plot(df_est['t'], errcdx, 'r', label='err_cdx')
    plt.plot(df_est['t'], errcdy, 'g', label='err_cdy')
    plt.plot(df_est['t'], errcdz, 'b', label='err_cdz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_vel_err.png')

    # Angular momentum
    fig = plt.figure('Angular momentum error')
    plt.title('Angular momentum error')
    plt.plot(df_est['t'], errLx, 'r', label='err_Lcx')
    plt.plot(df_est['t'], errLy, 'g', label='err_Lcy')
    plt.plot(df_est['t'], errLz, 'b', label='err_Lcz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'AM_err.png')

    # Estimated bias
    fig = plt.figure('PBC bias')
    plt.title('PBC bias')
    plt.plot(df_kfs['t'], df_kfs['bpx'], 'rx', label='bpx')
    plt.plot(df_kfs['t'], df_kfs['bpy'], 'gx', label='bpy')
    plt.plot(df_kfs['t'], df_kfs['bpz'], 'bx', label='bpz')
    plt.plot(df_gtr['t'], df_gtr['bpx'], 'r', label='bpx_gtr')
    plt.plot(df_gtr['t'], df_gtr['bpy'], 'g', label='bpy_gtr')
    plt.plot(df_gtr['t'], df_gtr['bpz'], 'b', label='bpz_gtr')

    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'bias_est.png')





# #############
# # Absolute plots
# #############
if ABSOLUTE_BASE:
    # Base position
    fig = plt.figure('Base position XYZ')
    plt.title('Base position XYZ')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['px'], 'r--', label='px_est')
    plt.plot(df_gtr['t'], df_gtr['px'], 'r', label='px_gtr')
    plt.plot(df_kfs['t'], df_kfs['px'], 'rx', label='px_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['py'], 'g--', label='py_est')
    plt.plot(df_gtr['t'], df_gtr['py'], 'g', label='py_gtr')
    plt.plot(df_kfs['t'], df_kfs['py'], 'gx', label='py_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['pz'], 'b--', label='pz_est')
    plt.plot(df_gtr['t'], df_gtr['pz'], 'b', label='pz_gtr')
    plt.plot(df_kfs['t'], df_kfs['pz'], 'bx', label='pz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'base_pos.png')

    # Base velocity
    fig = plt.figure('Base velocity XYZ')
    plt.title('Base velocity XYZ')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['vx'], 'r--', label='vx_est')
    plt.plot(df_gtr['t'], df_gtr['vx'], 'r', label='vx_gtr')
    plt.plot(df_kfs['t'], df_kfs['vx'], 'rx', label='vx_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['vy'], 'g--', label='cy_est')
    plt.plot(df_gtr['t'], df_gtr['vy'], 'g', label='cy_gtr')
    plt.plot(df_kfs['t'], df_kfs['vy'], 'gx', label='cy_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['vz'], 'b--', label='vz_est')
    plt.plot(df_gtr['t'], df_gtr['vz'], 'b', label='vz_gtr')
    plt.plot(df_kfs['t'], df_kfs['vz'], 'bx', label='vz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'base_vel.png')


if ABSOLUTE_COM_AM:
    # COM position
    fig = plt.figure('COM position')
    plt.title('COM position')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['cx'], 'r--', label='cx_est')
    plt.plot(df_est['t'], df_gtr['cx'], 'r', label='cx_gtr')
    plt.plot(df_kfs['t'], df_kfs['cx'], 'rx', label='cz_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['cy'], 'g--', label='cy_est')
    plt.plot(df_est['t'], df_gtr['cy'], 'g', label='cy_gtr')
    plt.plot(df_kfs['t'], df_kfs['cy'], 'bx', label='cz_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['cz'], 'b--', label='cz_est')
    plt.plot(df_est['t'], df_gtr['cz'], 'b', label='cz_gtr')
    plt.plot(df_kfs['t'], df_kfs['cz'], 'bx', label='cz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_pos.png')

    # COM velocity
    fig = plt.figure('COM velocity')
    plt.title('COM velocity')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['cdx'], 'r--', label='vx_est')
    plt.plot(df_est['t'], df_gtr['cdx'], 'r', label='vx_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdx'], 'rx', label='vz_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['cdy'], 'g--', label='vy_est')
    plt.plot(df_est['t'], df_gtr['cdy'], 'g', label='vy_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdy'], 'bx', label='vz_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['cdz'], 'b--', label='vz_est')
    plt.plot(df_est['t'], df_gtr['cdz'], 'b', label='vz_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdz'], 'bx', label='vz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_vel.png')

    # Angular momemtum
    fig = plt.figure('Angular Momentum')
    plt.title('Angular Momentum')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['Lx'], 'r--', label='Lx_est')
    plt.plot(df_est['t'], df_gtr['Lx'], 'r', label='Lx_gtr')
    plt.plot(df_kfs['t'], df_kfs['Lx'], 'rx', label='Lz_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['Ly'], 'g--', label='Ly_est')
    plt.plot(df_est['t'], df_gtr['Ly'], 'g', label='Ly_gtr')
    plt.plot(df_kfs['t'], df_kfs['Ly'], 'bx', label='Lz_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['Lz'], 'b--', label='Lz_est')
    plt.plot(df_est['t'], df_gtr['Lz'], 'b', label='Lz_gtr')
    plt.plot(df_kfs['t'], df_kfs['Lz'], 'bx', label='Lz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'AM.png')


# # Angular momentum 
# plt.figure()
# plt.plot(df_est['t'], df_est['Lx'], 'r--', label='Lx_est')
# plt.plot(df_est['t'], df_est['Ly'], 'g--', label='Ly_est')
# plt.plot(df_est['t'], df_est['Lz'], 'b--', label='Lz_est')
# plt.plot(df_gtr['t'], df_gtr['Lx'], 'r', label='Lx_gtr')
# plt.plot(df_gtr['t'], df_gtr['Ly'], 'g', label='Ly_gtr')
# plt.plot(df_gtr['t'], df_gtr['Lz'], 'b', label='Lz_gtr')
# plt.title('Angular momentum')
# plt.legend()


# # Wrench/kinematics plots
# plt.figure()
# plt.plot(df_wre.t, df_wre.pllx, 'r', label='pllx')
# plt.plot(df_wre.t, df_wre.plly, 'g', label='plly')
# plt.plot(df_wre.t, df_wre.pllz, 'b', label='pllz')
# plt.plot(df_wre.t, df_wre.prlx, 'r--', label='prlx')
# plt.plot(df_wre.t, df_wre.prly, 'g--', label='prly')
# plt.plot(df_wre.t, df_wre.prlz, 'b--', label='prlz')
# plt.legend()






############
# Wrench plots
############
# if WRENCH_PLOTS:
#     # forces
#     plt.figure('Force measurements')
#     plt.subplot(3,1,1)
#     plt.plot(df_wre.t, df_wre.f_lfx, 'g', label='f_lfx')
#     plt.plot(df_wre.t, df_wre.f_rfx, 'r', label='f_rfx')
#     plt.plot(df_wre.t, df_wre_tsid.f_lfx, 'g--', label='f_lfx_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.f_rfx, 'r--', label='f_rfx_tsid')
#     plt.subplot(3,1,2)
#     plt.plot(df_wre.t, df_wre.f_lfy, 'g', label='f_lfy')
#     plt.plot(df_wre.t, df_wre.f_rfy, 'r', label='f_rfy')
#     plt.plot(df_wre.t, df_wre_tsid.f_lfy, 'g--', label='f_lfy_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.f_rfy, 'r--', label='f_rfy_tsid')
#     plt.subplot(3,1,3)
#     plt.plot(df_wre.t, df_wre.f_lfz, 'g', label='f_lfz')
#     plt.plot(df_wre.t, df_wre.f_rfz, 'r', label='f_rfz')
#     plt.plot(df_wre.t, df_wre_tsid.f_lfz, 'g--', label='f_lfz_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.f_rfz, 'r--', label='f_rfz_tsid')
#     plt.title('Force measurements')
#     plt.legend()

#     # torques
#     plt.figure('Torque measurements')
#     plt.subplot(3,1,1)
#     plt.plot(df_wre.t, df_wre.tau_lfx, 'g', label='tau_llx')
#     plt.plot(df_wre.t, df_wre.tau_rfx, 'r', label='tau_rlx')
#     plt.plot(df_wre.t, df_wre_tsid.tau_lfx, 'g--', label='tau_lfx_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.tau_rfx, 'r--', label='tau_rfx_tsid')
#     plt.subplot(3,1,2)
#     plt.plot(df_wre.t, df_wre.tau_lfy, 'g', label='tau_lly')
#     plt.plot(df_wre.t, df_wre.tau_rfy, 'r', label='tau_rly')
#     plt.plot(df_wre.t, df_wre_tsid.tau_lfy, 'g--', label='tau_lfy_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.tau_rfy, 'r--', label='tau_rfy_tsid')
#     plt.subplot(3,1,3)
#     plt.plot(df_wre.t, df_wre.tau_lfz, 'g', label='tau_llz')
#     plt.plot(df_wre.t, df_wre.tau_rfz, 'r', label='tau_rlz')
#     plt.plot(df_wre.t, df_wre_tsid.tau_lfz, 'g--', label='tau_lfz_tsid')
#     plt.plot(df_wre.t, df_wre_tsid.tau_rfz, 'r--', label='tau_rfz_tsid')
#     plt.title('Torque measurements')
#     plt.legend()


if SAVE_FIGURES:
    print('Figures saved in '+SUB_DIR)

if PLOT_SHOW:
    plt.show()