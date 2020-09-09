#!/usr/bin/env python
# coding: utf-8

import os
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import pinocchio as pin
from experiment_naming import dirname_from_params

def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2))

SCALE_DIST = 0.1
MASS_DIST = True
PLOT = False

# Select plots to activate
RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_povcdl_estimation'
PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/mcapi_povcdl_estimation.yaml'
PARAM_FILE_BK = PARAM_FILE+'_bk'

FIG_DIR_PATH = 'figs/metrics/'

shutil.copyfile(PARAM_FILE, PARAM_FILE_BK)
with open(PARAM_FILE, 'r') as fr:
    params = yaml.safe_load(fr)
params['scale_dist']    = SCALE_DIST
params['mass_dist']     = MASS_DIST

SUB_DIR = dirname_from_params(params)
PATH = FIG_DIR_PATH + SUB_DIR + "/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

# vbc_is_dist, Iw_is_dist, Lgest_is_dist
experiments = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [0,1,1],
    [1,0,1],
    [1,1,1],
]

rmse_lst = []
maxe_lst = []

def stupid_true(v):
    return True if v else False


for i, things_to_dist in enumerate(experiments):
    print('Start experiment ', i)
    vbc_is_dist, Iw_is_dist, Lgest_is_dist = things_to_dist
    print(vbc_is_dist, Iw_is_dist, Lgest_is_dist)

    # change param file   
    params['vbc_is_dist']   = stupid_true(vbc_is_dist)
    params['Iw_is_dist']    = stupid_true(Iw_is_dist)
    params['Lgest_is_dist'] = stupid_true(Lgest_is_dist) 
    with open(PARAM_FILE, 'w') as fw:
        yaml.dump(params, fw)

    t1 = time.time()
    subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL)
    print('time: ', time.time()-t1)

    df_gtr = pd.read_csv('gtr.csv')
    df_est = pd.read_csv('est.csv')
    df_kfs = pd.read_csv('kfs.csv')
    df_wre = pd.read_csv('wre.csv')
    df_gtr_kfs = pd.DataFrame()
    for t in df_kfs['t']: 
        df_gtr_kfs = df_gtr_kfs.append(df_gtr.loc[np.isclose(df_gtr['t'], t)], ignore_index=True)


    # COMPUTE ERROR ARRAYS
    # Extract quaternions to compute orientation difference (xyzw reversed in constructor -> normal)
    quat_gtr_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_gtr['qx'], df_gtr['qy'], df_gtr['qz'], df_gtr['qw'])]
    quat_est_lst = [pin.Quaternion(qw, qx, qy, qz) for qx, qy, qz, qw in zip(df_est['qx'], df_est['qy'], df_est['qz'], df_est['qw'])]
    #  q_est - q_gtr =def log3(q_gtr.inv * q_est)
    quat_err_arr = np.array([pin.log3((q_gtr.inverse()*q_est).toRotationMatrix()) for q_gtr, q_est in zip(quat_gtr_lst, quat_est_lst)])
    errpx = df_est['px'] - df_gtr['px']
    errpy = df_est['py'] - df_gtr['py']
    errpz = df_est['pz'] - df_gtr['pz']
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
    errbpx = df_kfs['bpx'] - df_gtr_kfs['bpx']  # bias errors computed only at KFs (much sparser than the rest)
    errbpy = df_kfs['bpy'] - df_gtr_kfs['bpy']
    errbpz = df_kfs['bpz'] - df_gtr_kfs['bpz']

    # compute rmses
    rmse_px = rmse(errpx)
    rmse_py = rmse(errpy)
    rmse_pz = rmse(errpz)
    rmse_vx = rmse(errvx)
    rmse_vy = rmse(errvy)
    rmse_vz = rmse(errvz)
    rmse_cx = rmse(errcx)
    rmse_cy = rmse(errcy)
    rmse_cz = rmse(errcz)
    rmse_cdx = rmse(errcdx)
    rmse_cdy = rmse(errcdy)
    rmse_cdz = rmse(errcdz)
    rmse_Lx = rmse(errLx)
    rmse_Ly = rmse(errLy)
    rmse_Lz = rmse(errLz)
    rmse_bpx = rmse(errbpx)
    rmse_bpy = rmse(errbpy)
    rmse_bpz = rmse(errbpz)
    
    # compute max errors
    maxe_px = abs(errpx).max()
    maxe_py = abs(errpy).max()
    maxe_pz = abs(errpz).max()
    maxe_vx = abs(errvx).max()
    maxe_vy = abs(errvy).max()
    maxe_vz = abs(errvz).max()
    maxe_cx = abs(errcx).max()
    maxe_cy = abs(errcy).max()
    maxe_cz = abs(errcz).max()
    maxe_cdx = abs(errcdx).max()
    maxe_cdy = abs(errcdy).max()
    maxe_cdz = abs(errcdz).max()
    maxe_Lx = abs(errLx).max()
    maxe_Ly = abs(errLy).max()
    maxe_Lz = abs(errLz).max()
    maxe_bpx = abs(errbpx).max()
    maxe_bpy = abs(errbpy).max()
    maxe_bpz = abs(errbpz).max()

    rmse_lst.append([rmse_px,rmse_py,rmse_pz,rmse_vx,rmse_vy,rmse_vz,rmse_cx,rmse_cy,rmse_cz,rmse_cdx,rmse_cdy,rmse_cdz,rmse_Lx,rmse_Ly,rmse_Lz,rmse_bpx,rmse_bpy,rmse_bpz])
    maxe_lst.append([maxe_px, maxe_py, maxe_pz, maxe_vx, maxe_vy, maxe_vz, maxe_cx, maxe_cy, maxe_cz, maxe_cdx, maxe_cdy, maxe_cdz, maxe_Lx, maxe_Ly, maxe_Lz, maxe_bpx, maxe_bpy, maxe_bpz])
    
cols = ['px','py','pz','vx','vy','vz','cx','cy','cz','cdx','cdy','cdz','Lx','Ly','Lz','bpx','bpy','bpz']
index = []
for exp in experiments:
    idx = ''
    if exp[0]: idx += 'V'
    if exp[1]: idx += 'I'
    if exp[2]: idx += 'L'
    index.append(idx)
df_rmse = pd.DataFrame(rmse_lst, columns=cols, index=index)
df_maxe = pd.DataFrame(maxe_lst, columns=cols, index=index)
print(df_rmse)
print(df_maxe)

shutil.move(PARAM_FILE_BK, PARAM_FILE)

rmse_path = PATH+'rmse.csv'
maxe_path = PATH+'maxe.csv'
df_rmse.to_csv(rmse_path)
df_maxe.to_csv(maxe_path)


if PLOT:
    from mcapi_kpis_plot import plot_kpi_heatmaps
    plot_kpi_heatmaps(rmse_path, maxe_path, SCALE_DIST, MASS_DIST)