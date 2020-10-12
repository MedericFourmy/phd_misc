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
# struct = 'POV'
struct = 'POVCDL'

# Which figures
FBK = True
EST = True
PLOT_SHOW = True
SAVE_FIGURES = True

# Select plots to activate
if struct == 'POV':
    RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_pov_estimation'
elif struct == 'POVCDL':
    RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_povcdl_estimation'

PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/mcapi_povcdl_estimation.yaml'
RUN_SIMULATION =        True
FIG_DIR_PATH = 'figs/oneshot/'
SUB_DIR = dirname_from_params_path(PARAM_FILE, struct)
# SUB_DIR = 'test'
PATH = FIG_DIR_PATH + SUB_DIR + "/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

if RUN_SIMULATION:
    t1 = time.time()
    try:
        subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise(e)
    print('time: ', time.time()-t1)

df_gtr = pd.read_csv('gtr.csv')
df_est = pd.read_csv('est.csv')
df_fbk = pd.read_csv('fbk.csv')
df_kfs = pd.read_csv('kfs.csv')


# compute angle axis
quat_gtr_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_gtr['qx'], df_gtr['qy'], df_gtr['qz'], df_gtr['qw'])]
quat_est_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_est['qx'], df_est['qy'], df_est['qz'], df_est['qw'])]
quat_fbk_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_fbk['qx'], df_fbk['qy'], df_fbk['qz'], df_fbk['qw'])]
quat_kfs_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_kfs['qx'], df_kfs['qy'], df_kfs['qz'], df_kfs['qw'])]

o_gtr_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_gtr_lst])
o_est_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_est_lst])
o_fbk_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_fbk_lst])
o_kfs_arr = np.array([pin.log3(q.toRotationMatrix()) for q in quat_kfs_lst])
for i in range(3):
    xyz = 'xyz'[i]
    df_gtr['o'+xyz] = o_gtr_arr[:,i]
    df_est['o'+xyz] = o_est_arr[:,i]
    df_fbk['o'+xyz] = o_fbk_arr[:,i]
    df_kfs['o'+xyz] = o_kfs_arr[:,i]


def compute_err_dic(df_state, quat_state_lst, df_gtr, quat_gtr_lst, struct):
    err_dic = {}
    err_dic['px'] = df_state['px'] - df_gtr['px']
    err_dic['py'] = df_state['py'] - df_gtr['py']
    err_dic['pz'] = df_state['pz'] - df_gtr['pz']
    #  q_est - q_gtr =def log3(q_gtr.inv * q_est)
    quat_err_arr = np.array([pin.log3((q_gtr.inverse()*q_est).toRotationMatrix()) for q_gtr, q_est in zip(quat_gtr_lst, quat_state_lst)])
    quat_err_mag = np.apply_along_axis(np.linalg.norm, 1, quat_err_arr)
    err_dic['ox'] = quat_err_arr[:,0]
    err_dic['oy'] = quat_err_arr[:,1]
    err_dic['oz'] = quat_err_arr[:,2]
    err_dic['omag'] = quat_err_mag
    err_dic['vx'] =  df_state['vx'] - df_gtr['vx']
    err_dic['vy'] =  df_state['vy'] - df_gtr['vy']
    err_dic['vz'] =  df_state['vz'] - df_gtr['vz']
    if struct == 'POVCDL':
        err_dic['cx'] =  df_state['cx'] -  df_gtr['cx']
        err_dic['cy'] =  df_state['cy'] -  df_gtr['cy']
        err_dic['cz'] =  df_state['cz'] -  df_gtr['cz']
        err_dic['cdx'] = df_state['cdx'] - df_gtr['cdx']
        err_dic['cdy'] = df_state['cdy'] - df_gtr['cdy']
        err_dic['cdz'] = df_state['cdz'] - df_gtr['cdz']
        err_dic['Lcx'] =  df_state['Lcx'] -  df_gtr['Lcx']
        err_dic['Lcy'] =  df_state['Lcy'] -  df_gtr['Lcy']
        err_dic['Lcz'] =  df_state['Lcz'] -  df_gtr['Lcz']

    return err_dic


def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2))


def compute_rmse_errors(errd, struct):
    print('px err:  ', rmse(errd['px']))
    print('py err:  ', rmse(errd['py']))
    print('pz err:  ', rmse(errd['pz']))
    print('ox err:  ', rmse(errd['ox']))
    print('oy err:  ', rmse(errd['oy']))
    print('oz err:  ', rmse(errd['oz']))
    print('vx err:  ', rmse(errd['vx']))
    print('vy err:  ', rmse(errd['vy']))
    print('vz err:  ', rmse(errd['vz']))
    if struct == 'POVCDL':
        print('cx err:  ', rmse(errd['cx']))
        print('cy err:  ', rmse(errd['cy']))
        print('cz err:  ', rmse(errd['cz']))
        print('cdx err: ', rmse(errd['cdx']))
        print('cdy err: ', rmse(errd['cdy']))
        print('cdz err: ', rmse(errd['cdz']))
        print('Lcx err:  ', rmse(errd['cdx']))
        print('Lcy err:  ', rmse(errd['cdy']))
        print('Lcz err:  ', rmse(errd['cdz']))

# Compute error trajectories
errd_est = compute_err_dic(df_est, quat_est_lst, df_gtr, quat_gtr_lst, struct)
errd_fbk = compute_err_dic(df_fbk, quat_fbk_lst, df_gtr, quat_gtr_lst, struct)

print('EST rmse errors')
compute_rmse_errors(errd_est, struct)
print('FBK rmse errors')
compute_rmse_errors(errd_fbk, struct)

#############
# Error plots
#############
# base position
fig = plt.figure('Base position error')
plt.title('Base position error')
plt.plot(df_est['t'], errd_est['px'], 'r', label='err_px')
plt.plot(df_est['t'], errd_est['py'], 'g', label='err_py')
plt.plot(df_est['t'], errd_est['pz'], 'b', label='err_pz')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_pos_err.png')

# base orientation
fig = plt.figure('Orientation error')
plt.title('Orientation error')
plt.plot(df_est['t'], errd_est['ox'], 'r', label='err_orient_x')
plt.plot(df_est['t'], errd_est['oy'], 'g', label='err_orient_y')
plt.plot(df_est['t'], errd_est['oz'], 'b', label='err_orient_z')
plt.plot(df_est['t'], errd_est['omag'], 'y', label='err_orient_mag')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_orientation_err.png')

# base vel
fig = plt.figure('Base velocity error')
plt.title('Base velocity error')
plt.plot(df_est['t'], errd_est['vx'], 'r', label='err_vx')
plt.plot(df_est['t'], errd_est['vy'], 'g', label='err_vy')
plt.plot(df_est['t'], errd_est['vz'], 'b', label='err_vz')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_vel_err.png')

if struct == 'POVCDL':
    # CoM position
    fig = plt.figure('COM position error')
    plt.title('COM position error')
    plt.plot(df_est['t'], errd_est['cx'], 'r', label='err_cx')
    plt.plot(df_est['t'], errd_est['cy'], 'g', label='err_cy')
    plt.plot(df_est['t'], errd_est['cz'], 'b', label='err_cz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_pos_err.png')

    # CoM vel
    fig = plt.figure('COM velocity error')
    plt.title('COM velocity error')
    plt.plot(df_est['t'], errd_est['cdx'], 'r', label='err_cdx')
    plt.plot(df_est['t'], errd_est['cdy'], 'g', label='err_cdy')
    plt.plot(df_est['t'], errd_est['cdz'], 'b', label='err_cdz')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_vel_err.png')

    # Angular momentum
    fig = plt.figure('Angular momentum error')
    plt.title('Angular momentum error')
    plt.plot(df_est['t'], errd_est['Lcx'], 'r', label='err_Lcx')
    plt.plot(df_est['t'], errd_est['Lcy'], 'g', label='err_Lcy')
    plt.plot(df_est['t'], errd_est['Lcz'], 'b', label='err_Lcz')
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
# Base position
fig = plt.figure('Base position XYZ')
plt.title('Base position XYZ')
plt.subplot(3,1,1)
plt.plot(df_est['t'], df_est['px'], 'r--', label='px_est')
plt.plot(df_est['t'], df_fbk['px'], 'r:', label='px_fbk')
plt.plot(df_gtr['t'], df_gtr['px'], 'r',   label='px_gtr')
plt.plot(df_kfs['t'], df_kfs['px'], 'rx',  label='px_kf')
plt.subplot(3,1,2)
plt.plot(df_est['t'], df_est['py'], 'g--', label='py_est')
plt.plot(df_est['t'], df_fbk['py'], 'g:', label='py_fbk')
plt.plot(df_gtr['t'], df_gtr['py'], 'g',   label='py_gtr')
plt.plot(df_kfs['t'], df_kfs['py'], 'gx',  label='py_kf')
plt.subplot(3,1,3)
plt.plot(df_est['t'], df_est['pz'], 'b--', label='pz_est')
plt.plot(df_est['t'], df_fbk['pz'], 'b:', label='pz_fbk')
plt.plot(df_gtr['t'], df_gtr['pz'], 'b',   label='pz_gtr')
plt.plot(df_kfs['t'], df_kfs['pz'], 'bx',  label='pz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_pos.png')


# Base orientation
fig = plt.figure('Base orientation AngleAxis World frame')
plt.title('Base orientation XYZ')
plt.subplot(3,1,1)
plt.plot(df_est['t'], df_est['ox'], 'r--', label='ox_est')
plt.plot(df_est['t'], df_fbk['ox'], 'r:', label='ox_fbk')
plt.plot(df_gtr['t'], df_gtr['ox'], 'r',   label='ox_gtr')
plt.plot(df_kfs['t'], df_kfs['ox'], 'rx',  label='ox_kf')
plt.subplot(3,1,2)
plt.plot(df_est['t'], df_est['oy'], 'g--', label='oy_est')
plt.plot(df_est['t'], df_fbk['oy'], 'g:', label='oy_fbk')
plt.plot(df_gtr['t'], df_gtr['oy'], 'g',   label='oy_gtr')
plt.plot(df_kfs['t'], df_kfs['oy'], 'gx',  label='oy_kf')
plt.subplot(3,1,3)
plt.plot(df_est['t'], df_est['oz'], 'b--', label='oz_est')
plt.plot(df_est['t'], df_fbk['oz'], 'b:', label='oz_fbk')
plt.plot(df_gtr['t'], df_gtr['oz'], 'b',   label='oz_gtr')
plt.plot(df_kfs['t'], df_kfs['oz'], 'bx',  label='oz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_angleaxis.png')


# Base velocity
fig = plt.figure('Base velocity XYZ')
plt.title('Base velocity XYZ')
plt.subplot(3,1,1)
plt.plot(df_est['t'], df_est['vx'], 'r--', label='vx_est')
plt.plot(df_est['t'], df_fbk['vx'], 'r:',  label='vx_fbk')
plt.plot(df_gtr['t'], df_gtr['vx'], 'r',   label='vx_gtr')
plt.plot(df_kfs['t'], df_kfs['vx'], 'rx',  label='vx_kf')
plt.subplot(3,1,2)
plt.plot(df_est['t'], df_est['vy'], 'g--', label='vy_est')
plt.plot(df_est['t'], df_fbk['vy'], 'g:',  label='vy_fbk')
plt.plot(df_gtr['t'], df_gtr['vy'], 'g',   label='vy_gtr')
plt.plot(df_kfs['t'], df_kfs['vy'], 'gx',  label='vy_kf')
plt.subplot(3,1,3)
plt.plot(df_est['t'], df_est['vz'], 'b--', label='vz_est')
plt.plot(df_est['t'], df_fbk['vz'], 'b:',  label='vz_fbk')
plt.plot(df_gtr['t'], df_gtr['vz'], 'b',   label='vz_gtr')
plt.plot(df_kfs['t'], df_kfs['vz'], 'bx',  label='vz_kf')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'base_vel.png')

# Estimated bias
fig = plt.figure('IMU bias')
plt.title('IMU bias')
plt.plot(df_kfs['t'], df_kfs['bax'], 'rx', label='bax_est')
plt.plot(df_kfs['t'], df_kfs['bay'], 'gx', label='bay_est')
plt.plot(df_kfs['t'], df_kfs['baz'], 'bx', label='baz_est')
plt.plot(df_kfs['t'], df_kfs['bwx'], 'r.', label='bwx_est')
plt.plot(df_kfs['t'], df_kfs['bwy'], 'g.', label='bwy_est')
plt.plot(df_kfs['t'], df_kfs['bwz'], 'b.', label='bwz_est')
plt.legend()
if SAVE_FIGURES: fig.savefig(PATH+'imu_bias_est.png')


if struct == 'POVCDL':
    # COM position
    fig = plt.figure('COM position')
    plt.title('COM position')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['cx'], 'r--', label='cx_est')
    plt.plot(df_est['t'], df_fbk['cx'], 'r:',  label='cx_fbk')
    plt.plot(df_est['t'], df_gtr['cx'], 'r', label='cx_gtr')
    plt.plot(df_kfs['t'], df_kfs['cx'], 'rx', label='cx_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['cy'], 'g--', label='cy_est')
    plt.plot(df_est['t'], df_fbk['cy'], 'g:',  label='cy_fbk')
    plt.plot(df_est['t'], df_gtr['cy'], 'g', label='cy_gtr')
    plt.plot(df_kfs['t'], df_kfs['cy'], 'gx', label='cy_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['cz'], 'b--', label='cz_est')
    plt.plot(df_est['t'], df_fbk['cz'], 'b:',  label='cz_fbk')
    plt.plot(df_est['t'], df_gtr['cz'], 'b', label='cz_gtr')
    plt.plot(df_kfs['t'], df_kfs['cz'], 'bx', label='cz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_pos.png')

    # COM velocity
    fig = plt.figure('COM velocity')
    plt.title('COM velocity')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['cdx'], 'r--', label='cdx_est')
    plt.plot(df_est['t'], df_fbk['cdx'], 'r:',  label='cdx_fbk')
    plt.plot(df_est['t'], df_gtr['cdx'], 'r', label='cdx_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdx'], 'rx', label='cdx_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['cdy'], 'g--', label='cdy_est')
    plt.plot(df_est['t'], df_fbk['cdy'], 'g:',  label='cdy_fbk')
    plt.plot(df_est['t'], df_gtr['cdy'], 'g', label='cdy_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdy'], 'gx', label='cdy_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['cdz'], 'b--', label='cdz_est')
    plt.plot(df_est['t'], df_fbk['cdz'], 'b:',  label='cdz_fbk')
    plt.plot(df_est['t'], df_gtr['cdz'], 'b', label='cdz_gtr')
    plt.plot(df_kfs['t'], df_kfs['cdz'], 'bx', label='cdz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'COM_vel.png')

    # Angular momemtum
    fig = plt.figure('Angular Momentum')
    plt.title('Angular Momentum')
    plt.subplot(3,1,1)
    plt.plot(df_est['t'], df_est['Lcx'], 'r--', label='Lcx_est')
    plt.plot(df_est['t'], df_fbk['Lcx'], 'r:',  label='Lcx_fbk')
    plt.plot(df_est['t'], df_gtr['Lcx'], 'r', label='Lcx_gtr')
    plt.plot(df_kfs['t'], df_kfs['Lcx'], 'rx', label='Lcz_kf')
    plt.subplot(3,1,2)
    plt.plot(df_est['t'], df_est['Lcy'], 'g--', label='Lcy_est')
    plt.plot(df_est['t'], df_fbk['Lcy'], 'g:',  label='Lcy_fbk')
    plt.plot(df_est['t'], df_gtr['Lcy'], 'g', label='Lcy_gtr')
    plt.plot(df_kfs['t'], df_kfs['Lcy'], 'gx', label='Lcz_kf')
    plt.subplot(3,1,3)
    plt.plot(df_est['t'], df_est['Lcz'], 'b--', label='Lcz_est')
    plt.plot(df_est['t'], df_fbk['Lcz'], 'b:',  label='Lcz_fbk')
    plt.plot(df_est['t'], df_gtr['Lcz'], 'b', label='Lcz_gtr')
    plt.plot(df_kfs['t'], df_kfs['Lcz'], 'bx', label='Lcz_kf')
    plt.legend()
    if SAVE_FIGURES: fig.savefig(PATH+'AM.png')




if SAVE_FIGURES:
    print('Figures saved in '+SUB_DIR)

if PLOT_SHOW:
    plt.show()
