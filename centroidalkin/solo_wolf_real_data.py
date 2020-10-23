#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import pinocchio as pin
from experiment_naming import dirname_from_params_path

# CHOOSE which problem you want to solve
struct = 'POV'
# struct = 'POVCDL'

# Which figures
EST = True
FBK = False
KF = False
PLOT_SHOW = True
SAVE_FIGURES = True


# Select plots to activate
if struct == 'POV':
    RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_real_pov_estimation'
elif struct == 'POVCDL':
    RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_real_povcdl_estimation'

PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/solo_real_estimation.yaml'
RUN_SIMULATION = True
FIG_DIR_PATH = 'figs/oneshotreal/'
SUB_DIR = dirname_from_params_path(PARAM_FILE, struct)
# SUB_DIR = 'test'
PATH = FIG_DIR_PATH + SUB_DIR + "/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

if RUN_SIMULATION:
    t1 = time.time()
    try:
        print('Running ', RUN_FILE)
        subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise(e)
    print('time: ', time.time()-t1)

df_est = pd.read_csv('est.csv')
df_fbk = pd.read_csv('fbk.csv')
df_kfs = pd.read_csv('kfs.csv')


# compute angle axis
quat_est_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_est['qx'], df_est['qy'], df_est['qz'], df_est['qw'])]
quat_fbk_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_fbk['qx'], df_fbk['qy'], df_fbk['qz'], df_fbk['qw'])]
quat_kfs_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_kfs['qx'], df_kfs['qy'], df_kfs['qz'], df_kfs['qw'])]

o_est_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_est_lst])
o_fbk_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_fbk_lst])
o_kfs_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_kfs_lst])
for i in range(3):
    xyz = 'xyz'[i]
    df_est['o'+xyz] = o_est_arr[:,i]
    df_fbk['o'+xyz] = o_fbk_arr[:,i]
    df_kfs['o'+xyz] = o_kfs_arr[:,i]







# #############
# # Absolute plots
# #############
# Base position
fig = plt.figure('Base position XYZ')
plt.title('Base position XYZ')
plt.subplot(3,1,1)
if EST: plt.plot(df_est['t'], df_est['px'], 'r--', label='px_est')
if FBK: plt.plot(df_est['t'], df_fbk['px'], 'r:', label='px_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['px'], 'rx',  label='px_kf')
plt.legend()
plt.subplot(3,1,2)
if EST: plt.plot(df_est['t'], df_est['py'], 'g--', label='py_est')
if FBK: plt.plot(df_est['t'], df_fbk['py'], 'g:', label='py_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['py'], 'gx',  label='py_kf')
plt.legend()
plt.subplot(3,1,3)
if EST: plt.plot(df_est['t'], df_est['pz'], 'b--', label='pz_est')
if FBK: plt.plot(df_est['t'], df_fbk['pz'], 'b:', label='pz_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['pz'], 'bx',  label='pz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_pos.png')


# Base orientation
fig = plt.figure('Base orientation AngleAxis World frame')
plt.title('Base orientation XYZ')
plt.subplot(3,1,1)
if EST: plt.plot(df_est['t'], df_est['ox'], 'r--', label='ox_est')
if FBK: plt.plot(df_est['t'], df_fbk['ox'], 'r:', label='ox_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['ox'], 'rx',  label='ox_kf')
plt.legend()
plt.subplot(3,1,2)
if EST: plt.plot(df_est['t'], df_est['oy'], 'g--', label='oy_est')
if FBK: plt.plot(df_est['t'], df_fbk['oy'], 'g:', label='oy_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['oy'], 'gx',  label='oy_kf')
plt.legend()
plt.subplot(3,1,3)
if EST: plt.plot(df_est['t'], df_est['oz'], 'b--', label='oz_est')
if FBK: plt.plot(df_est['t'], df_fbk['oz'], 'b:', label='oz_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['oz'], 'bx',  label='oz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_angleaxis.png')

# Base orientation
fig = plt.figure('Base orientation Quaternion World frame')
plt.title('Base orientation XYZ')
plt.subplot(4,1,1)
if EST: plt.plot(df_est['t'], df_est['qx'], 'r--', label='qx_est')
if FBK: plt.plot(df_est['t'], df_fbk['qx'], 'r:', label='qx_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['qx'], 'rx',  label='qx_kf')
plt.legend()
plt.subplot(4,1,2)
if EST: plt.plot(df_est['t'], df_est['qy'], 'g--', label='qy_est')
if FBK: plt.plot(df_est['t'], df_fbk['qy'], 'g:', label='qy_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['qy'], 'gx',  label='qy_kf')
plt.legend()
plt.subplot(4,1,3)
if EST: plt.plot(df_est['t'], df_est['qz'], 'b--', label='qz_est')
if FBK: plt.plot(df_est['t'], df_fbk['qz'], 'b:', label='qz_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['qz'], 'bx',  label='qz_kf')
plt.legend()
plt.subplot(4,1,4)
if EST: plt.plot(df_est['t'], df_est['qw'], 'k--', label='qw_est')
if FBK: plt.plot(df_est['t'], df_fbk['qw'], 'k:', label='qw_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['qw'], 'kx',  label='qw_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_angleaxis.png')


# Base velocity
fig = plt.figure('Base velocity XYZ')
plt.title('Base velocity XYZ')
plt.subplot(3,1,1)
if EST: plt.plot(df_est['t'], df_est['vx'], 'r--', label='vx_est')
if FBK: plt.plot(df_est['t'], df_fbk['vx'], 'r:',  label='vx_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['vx'], 'rx',  label='vx_kf')
plt.legend()
plt.subplot(3,1,2)
if EST: plt.plot(df_est['t'], df_est['vy'], 'g--', label='vy_est')
if FBK: plt.plot(df_est['t'], df_fbk['vy'], 'g:',  label='vy_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['vy'], 'gx',  label='vy_kf')
plt.legend()
plt.subplot(3,1,3)
if EST: plt.plot(df_est['t'], df_est['vz'], 'b--', label='vz_est')
if FBK: plt.plot(df_est['t'], df_fbk['vz'], 'b:',  label='vz_fbk')
if KF:  plt.plot(df_kfs['t'], df_kfs['vz'], 'bx',  label='vz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_vel.png')

# Estimated bias
fig = plt.figure('IMU bias')
plt.subplot(2,1,1)
plt.plot(df_kfs['t'], df_kfs['bax'], 'rx', label='acc_x')
plt.plot(df_kfs['t'], df_kfs['bay'], 'gx', label='acc_y')
plt.plot(df_kfs['t'], df_kfs['baz'], 'bx', label='acc_z')
plt.legend()
plt.subplot(2,1,2)
plt.plot(df_kfs['t'], df_kfs['bwx'], 'r.', label='gyro_x')
plt.plot(df_kfs['t'], df_kfs['bwy'], 'g.', label='gyro_y')
plt.plot(df_kfs['t'], df_kfs['bwz'], 'b.', label='gyro_z')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'imu_bias_est.png')


if struct == 'POVCDL':
    # COM position
    fig = plt.figure('COM position')
    plt.title('COM position')
    plt.subplot(3,1,1)
    if EST: plt.plot(df_est['t'], df_est['cx'], 'r--', label='cx_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cx'], 'r:',  label='cx_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cx'], 'rx', label='cx_kf')
    plt.legend()
    plt.subplot(3,1,2)
    if EST: plt.plot(df_est['t'], df_est['cy'], 'g--', label='cy_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cy'], 'g:',  label='cy_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cy'], 'gx', label='cy_kf')
    plt.legend()
    plt.subplot(3,1,3)
    if EST: plt.plot(df_est['t'], df_est['cz'], 'b--', label='cz_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cz'], 'b:',  label='cz_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cz'], 'bx', label='cz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_pos.png')

    # COM velocity
    fig = plt.figure('COM velocity')
    plt.title('COM velocity')
    plt.subplot(3,1,1)
    if EST: plt.plot(df_est['t'], df_est['cdx'], 'r--', label='cdx_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cdx'], 'r:',  label='cdx_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cdx'], 'rx', label='cdx_kf')
    plt.legend()
    plt.subplot(3,1,2)
    if EST: plt.plot(df_est['t'], df_est['cdy'], 'g--', label='cdy_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cdy'], 'g:',  label='cdy_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cdy'], 'gx', label='cdy_kf')
    plt.legend()
    plt.subplot(3,1,3)
    if EST: plt.plot(df_est['t'], df_est['cdz'], 'b--', label='cdz_est')
    if FBK: plt.plot(df_est['t'], df_fbk['cdz'], 'b:',  label='cdz_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['cdz'], 'bx', label='cdz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_vel.png')

    # Angular momemtum
    fig = plt.figure('Angular Momentum')
    plt.title('Angular Momentum')
    plt.subplot(3,1,1)
    if EST: plt.plot(df_est['t'], df_est['Lx'], 'r--', label='Lcx_est')
    if FBK: plt.plot(df_est['t'], df_fbk['Lx'], 'r:',  label='Lcx_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['Lx'], 'rx', label='Lcz_kf')
    plt.legend()
    plt.subplot(3,1,2)
    if EST: plt.plot(df_est['t'], df_est['Ly'], 'g--', label='Lcy_est')
    if FBK: plt.plot(df_est['t'], df_fbk['Ly'], 'g:',  label='Lcy_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['Ly'], 'gx', label='Lcz_kf')
    plt.legend()
    plt.subplot(3,1,3)
    if EST: plt.plot(df_est['t'], df_est['Lz'], 'b--', label='Lcz_est')
    if FBK: plt.plot(df_est['t'], df_fbk['Lz'], 'b:',  label='Lcz_fbk')
    if KF:  plt.plot(df_kfs['t'], df_kfs['Lz'], 'bx', label='Lcz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'AM.png')




if SAVE_FIGURES:
    print('Figures saved in '+SUB_DIR)

if PLOT_SHOW:
    plt.show()