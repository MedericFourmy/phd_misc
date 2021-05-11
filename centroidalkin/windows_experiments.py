#!/usr/bin/env python
# coding: utf-8

"""
Goal: find a satisfying operational point for different trajectories
by varying the sliding window parameters
"""

import os
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import subprocess
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pinocchio as pin
from experiment_naming import dirname_from_params

CLOSE = False
SHOW = True

def nb_possibilities(lst):
    param_nb_lst = [len(l) for l in lst]
    return np.prod(param_nb_lst)

def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2, axis=0))

def create_bak_file_and_get_params(path):
    path_bak = path+'_bak'
    shutil.copyfile(path, path_bak)
    with open(path, 'r') as fr:
        params = yaml.safe_load(fr)
    return params

def restore_initial_file(str):
    path_bak = path+'_bak'
    # shutil.move(path_bak, path)
    shutil.copyfile(path_bak, path)

def diff_shift(arr):
    diff = np.roll(arr, -1, axis=0) - arr
    # last element/line will be huge since roll wraps around the array
    # -> just copy the n_last - 1
    diff[-1,:] = diff[-2,:]
    return diff


# executable and param files paths
# RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_real_pov_estimation'
RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_mocap_imu'
PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/solo_real_estimation.yaml'
SENSOR_IMU_PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/sensor_imu_solo12.yaml'
PROC_IMU_PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/processor_imu_solo12.yaml'
TREE_PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/tree_manager.yaml'

FIG_DIR_PATH = 'figs/window_experiments/'

shutil.rmtree(FIG_DIR_PATH, ignore_errors=True)

params_proc_imu = create_bak_file_and_get_params(PROC_IMU_PARAM_FILE)
params_sensor_imu = create_bak_file_and_get_params(SENSOR_IMU_PARAM_FILE)
params = create_bak_file_and_get_params(PARAM_FILE)
params_tree = create_bak_file_and_get_params(TREE_PARAM_FILE)

if not os.path.exists(FIG_DIR_PATH):
    os.makedirs(FIG_DIR_PATH)

# Choose TRAJECTORY directly here
params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/SinStamping_Corrected_09_12_2020/data_2020_12_09_17_54_format.npz'  # sin
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/SinStamping_Corrected_09_12_2020/data_2020_12_09_17_56_format.npz'  # stamping
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Walk_17_12_2020/data_2020_12_17_14_25_format.npz'  # point feet walk with corrected kinematics
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Walk_17_12_2020/data_2020_12_17_14_29_format.npz'  # point feet walk with corrected kinematics

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Mesures_Contact/data_2021_01_19_17_11_format.npz'

# Hand held
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_00_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_01_format.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_03_format_short.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_03_format_longer.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_03_format_much_longer.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_04_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_07_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_08_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_15_format.npz'


# new experiments
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_09_format.npz'  # Standing still (5s), mocap 200Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_10_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_16_format.npz'  # Moving up (5s), mocap 200Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_17_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_25_format.npz'  # Standing still (5s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_26_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_29_format.npz'  # Moving up (5s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_30_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_31_format.npz'  # Moving up->front->down (10s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_32_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_54_format.npz'  # Moving up then random movements (15s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_56_format.npz'  # //
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_57_format.npz'  # //
params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_59_format.npz'  # Already in air, random movements (15s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_16_03_format.npz'  # //


# other main params
params['unfix_extr_sensor_pose'] = False

# MOCAP
params['std_pose_p'] = 0.0005
params['std_pose_o_deg'] = 1

# IMU params
params_sensor_imu['motion_variances']['a_noise'] =       0.02     # standard deviation of Acceleration noise (same for all the axis) in m/s2
params_sensor_imu['motion_variances']['w_noise'] =       0.03    # standard deviation of Gyroscope noise (same for all the axis) in rad/sec
params_sensor_imu['motion_variances']['ab_rate_stdev'] = 1e-6       # m/s2/sqrt(s)           
params_sensor_imu['motion_variances']['wb_rate_stdev'] = 1e-6    # rad/s/sqrt(s)
# PRIOR IMU
# params['bias_imu_prior'] = [ 0.01323934,  0.10560589, -0.00157562,  0.00242635,  0.00111252, -0.01004376]
params['bias_imu_prior'] = [0]*6
params['std_abs_bias_acc'] =  100000
params['std_abs_bias_gyro'] = 100000

params['dt'] = 2e-3
params['max_t'] = 15





# trial 1
# max_t_kf_lst = [0.005, 0.01, 0.05, 0.1]
# KF_nb_lst = [10, 30, 50, 70, 90]
# time_shift_mocap_lst = [2]

# # trial 2 (SIN)
# # max_t_kf_lst = [0.05, 0.06, 0.07, 0.08, 0.09]
# max_t_kf_lst = [0.08, 0.09, 0.10, 0.11]
# KF_nb_lst = [10, 20, 30]
# time_shift_mocap_lst = [2]

# max_t_kf_lst = [0.05]
# KF_nb_lst = [5, 100]
# time_shift_mocap_lst = [1]


max_t_kf_lst = [0.19999]
# KF_nb_lst = [30, 50, 100, 150, 200, 300, 400, 500, 10000000000]
KF_nb_lst = [100000000000]
# time_shift_mocap_lst = [-0.01, -0.005, -0.001, 0, 0.001, 0.002, 0.005, 0.01, 0.1]
# time_shift_mocap_lst = [-0.002, -0.001, 0, 0.001, 0.002]
# time_shift_mocap_lst = [0, 0.09]
time_shift_mocap_lst = [0]

max_t_kf_idx_lst = np.arange(len(max_t_kf_lst))
KF_nb_idx_lst = np.arange(len(KF_nb_lst))
time_shift_mocap_idx_lst = np.arange(len(time_shift_mocap_lst))



possibs = nb_possibilities([max_t_kf_lst, KF_nb_lst, time_shift_mocap_lst])
print('Combinations to evaluate: ', possibs)


RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out.npz'
params['out_npz_file_path'] = RESULTS

rmse_pos_arr = np.zeros((len(max_t_kf_lst), len(KF_nb_lst), len(time_shift_mocap_lst)))
rmse_vel_arr = np.zeros((len(max_t_kf_lst), len(KF_nb_lst), len(time_shift_mocap_lst)))
compute_time_arr = np.zeros((len(max_t_kf_lst), len(KF_nb_lst), len(time_shift_mocap_lst)))

for i, (max_t_kf_idx, KF_nb_idx, time_shift_mocap_idx) in enumerate(itertools.product(max_t_kf_idx_lst, KF_nb_idx_lst, time_shift_mocap_idx_lst)):
    max_t_kf = max_t_kf_lst[max_t_kf_idx]
    KF_nb = KF_nb_lst[KF_nb_idx]
    time_shift_mocap = time_shift_mocap_lst[time_shift_mocap_idx]

    params['time_shift_mocap'] = time_shift_mocap
    with open(PARAM_FILE, 'w') as fw: yaml.dump(params, fw)

    with open(SENSOR_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_sensor_imu, fw)

    params_proc_imu['keyframe_vote']['max_time_span'] = max_t_kf
    with open(PROC_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_proc_imu, fw)

    params_tree['config']['problem']['tree_manager']['n_frames'] = KF_nb
    with open(TREE_PARAM_FILE, 'w') as fw: yaml.dump(params_tree, fw)



    # run executable
    t1 = time.time()
    # subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL)
    subprocess.run(RUN_FILE)
    compute_time = time.time()-t1
    compute_time_arr[max_t_kf_idx, KF_nb_idx, time_shift_mocap_idx] = compute_time 
    print(i, ':', compute_time)
    
    config = {
        'max_t_kf': max_t_kf,
        'KF_nb': KF_nb,
        'time_shift_mocap': time_shift_mocap
    }
    with open(FIG_DIR_PATH+'conf_{}.yaml'.format(i), 'w') as fw: yaml.dump(config, fw)

    # load raw results and compute rmses
    arr_dic = np.load(RESULTS)
    t_arr = arr_dic['t']
    o_p_ob_arr = arr_dic['o_p_ob']
    o_q_b_arr = arr_dic['o_q_b']
    o_v_ob_arr = arr_dic['o_v_ob']
    o_p_ob_fbk_arr = arr_dic['o_p_ob_fbk']
    o_q_b_fbk_arr = arr_dic['o_q_b_fbk']
    o_v_ob_fbk_arr = arr_dic['o_v_ob_fbk']

    o_p_ob_diff = diff_shift(o_p_ob_arr)
    o_v_ob_diff = diff_shift(o_v_ob_arr)
    # rmse_pos_arr[max_t_kf_idx, KF_nb_idx, time_shift_mocap_idx] = rmse(o_p_ob_diff).mean()
    # rmse_vel_arr[max_t_kf_idx, KF_nb_idx, time_shift_mocap_idx] = rmse(o_v_ob_diff).mean()

    # retrieve ground truth for comparison
    w_p_wm_arr = arr_dic['w_p_wm']
    w_q_m_arr = arr_dic['w_q_m']
    w_v_wm_arr = arr_dic['w_v_wm']



    w_p_wm_init = w_p_wm_arr[0,:]
    w_R_m_init = pin.Quaternion(w_q_m_arr[0,:].reshape((4,1))).toRotationMatrix()
    # w_T_m_init = pin.SE3(w_R_m_init, w_p_wm_init)
    w_T_m_init = pin.SE3.Identity()

    o_p_ob_init = o_p_ob_arr[0,:]
    o_R_b_init = pin.Quaternion(o_q_b_arr[0,:].reshape((4,1))).toRotationMatrix()
    # o_T_b_init = pin.SE3(o_R_b_init, o_p_ob_init)
    o_T_b_init = pin.SE3.Identity()

    o_p_ob_fbk_init = o_p_ob_fbk_arr[0,:]
    o_R_b_fbk_init = pin.Quaternion(o_q_b_fbk_arr[0,:].reshape((4,1))).toRotationMatrix()
    # o_T_b_fbk_init = pin.SE3(o_R_b_fbk_init, o_p_ob_fbk_init)
    o_T_b_fbk_init = pin.SE3.Identity()

    # transform estimated trajectories in mocap frame
    w_T_o = w_T_m_init * o_T_b_init.inverse()
    w_p_wb_arr = np.array([w_T_o.act(o_p_ob) for o_p_ob in o_p_ob_arr])
    w_v_wb_arr = np.array([w_T_o.rotation@o_v_ob for o_v_ob in o_v_ob_arr])
    w_T_o_fbk = w_T_m_init * o_T_b_init.inverse()
    w_p_wb_fbk_arr = np.array([w_T_o_fbk.act(o_p_ob) for o_p_ob in o_p_ob_fbk_arr])
    w_v_wb_fbk_arr = np.array([w_T_o_fbk.rotation@o_v_ob for o_v_ob in o_v_ob_fbk_arr])


    # biases
    imu_bias = arr_dic['imu_bias']
    imu_bias_fbk = arr_dic['imu_bias_fbk']
    extr_mocap_fbk = arr_dic['extr_mocap_fbk']

    # covariances
    Nsig = 2
    tkf_arr = arr_dic['tkf']
    Nkf = len(tkf_arr)
    Qp = arr_dic['Qp']
    Qo = arr_dic['Qo']
    Qv = arr_dic['Qv']
    # Qbi = arr_dic['Qbi']
    Qbi = arr_dic['Qbi_fbk']
    Qm = arr_dic['Qm']
    envel_p =  Nsig*np.sqrt(Qp)
    envel_o =  Nsig*np.sqrt(Qo)
    envel_v =  Nsig*np.sqrt(Qv)
    envel_bi = Nsig*np.sqrt(Qbi)
    envel_m =  Nsig*np.sqrt(Qm)

    # bias cov can explode at the beginning
    clip = np.array(3*[1.5] + 3*[0.3])
    envel_bi = np.clip(envel_bi, np.zeros((Nkf, 6)), clip)

    # factor errors
    fac_imu_err = arr_dic['fac_imu_err']
    fac_pose_err = arr_dic['fac_pose_err']
    
    # # bias residuals
    # ab_rate_stdev = 0.0001
    # wb_rate_stdev = 0.0001
    # sqrt_A_r_dt_inv = 1/(ab_rate_stdev * np.sqrt(max_t_kf))
    # sqrt_W_r_dt_inv = 1/(wb_rate_stdev * np.sqrt(max_t_kf))
    # bias_drift_error = imu_bias - np.roll(imu_bias, 1, axis=0)
    # bias_drift_error[0,:] = bias_drift_error[1,:]
    # bias_acc_drift_res = sqrt_A_r_dt_inv*bias_drift_error[:,:3]
    # bias_gyr_drift_res = sqrt_W_r_dt_inv*bias_drift_error[:,3:6]





    print('imu_bias END:', imu_bias[-1,:])
    print('origin quaternion END:', o_q_b_arr[0,:])

    

    #######################
    # TRAJECTORY EST VS GTR
    #######################
    fig = plt.figure('Position est vs mocap'+str(i))
    plt.title('Position est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_p_wb_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_p_wb_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, w_p_wm_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('P (m)')
    plt.legend()
    # plt.savefig(FIG_DIR_PATH+'pos_{}.png'.format(i))
    plt.savefig(FIG_DIR_PATH+'pos_{}.eps'.format(i), format='eps')
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Velocity est vs mocap'+str(i))
    plt.title('Velocity est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_v_wb_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_v_wb_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, w_v_wm_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)


    fig = plt.figure('Orientation est vs mocap'+str(i))
    plt.title('Orientation est vs mocap\n{}'.format(config))
    for i in range(4):
        plt.plot(t_arr, o_q_b_arr[:,i], 'rgbk'[i], label='est')
        plt.plot(t_arr, o_q_b_fbk_arr[:,i], 'rgbk'[i]+'.', label='fbk')
        plt.plot(t_arr, w_q_m_arr[:,i], 'rgbk'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('Q')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)



    #############
    # ERROR plots
    #############
    fig = plt.figure('Position error and covariances'+str(i))
    plt.title('Position est vs mocap\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.plot(t_arr, w_p_wb_arr[:,k] - w_p_wm_arr[:,k], 'b', label='est')
        plt.plot(t_arr, w_p_wb_fbk_arr[:,k] - w_p_wm_arr[:,k], 'b.', label='fbk')
        plt.plot(tkf_arr,  envel_p[:,k], 'k', label='cov')
        plt.plot(tkf_arr, -envel_p[:,k], 'k', label='cov')
        plt.legend()
    plt.savefig(FIG_DIR_PATH+'pos_err_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)
    
    fig = plt.figure('Velocity error and covariances'+str(i))
    plt.title('Velocity error and covariances\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.plot(t_arr, w_v_wb_arr[:,k] - w_v_wm_arr[:,k], 'b', label='est')
        plt.plot(t_arr, w_v_wb_fbk_arr[:,k] - w_v_wm_arr[:,k], 'b.', label='fbk')
        plt.plot(tkf_arr,  envel_v[:,k], 'k', label='cov')
        plt.plot(tkf_arr, -envel_v[:,k], 'k', label='cov')
        plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_err_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)



    ###############
    # FACTOR ERRORS
    ###############
    fig = plt.figure('Factor IMU err'+str(i))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.title('POV'[k])
        plt.plot(tkf_arr, fac_imu_err[:,0+3*k], 'r')
        plt.plot(tkf_arr, fac_imu_err[:,1+3*k], 'g')
        plt.plot(tkf_arr, fac_imu_err[:,2+3*k], 'b')
    plt.savefig(FIG_DIR_PATH+'fac_imu_errors_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)

    # fig = plt.figure('Factor IMU bias res')
    # plt.subplot(2,1,1)
    # plt.title('Accelerometer bias res')
    # plt.plot(t_arr, bias_acc_drift_res[:,0], 'r')
    # plt.plot(t_arr, bias_acc_drift_res[:,1], 'g')
    # plt.plot(t_arr, bias_acc_drift_res[:,2], 'b')
    # plt.subplot(2,1,2)
    # plt.title('Gyro bias res')
    # plt.plot(t_arr, bias_gyr_drift_res[:,0], 'r')
    # plt.plot(t_arr, bias_gyr_drift_res[:,1], 'g')
    # plt.plot(t_arr, bias_gyr_drift_res[:,2], 'b')
    # plt.savefig(FIG_DIR_PATH+'fac_bias_drift_res_{}.png'.format(i))
    # if CLOSE: plt.close(fig=fig)

    # fig = plt.figure('Factor IMU bias error')
    # plt.subplot(2,1,1)
    # plt.title('Accelerometer bias error')
    # plt.plot(t_arr, bias_drift_error[:,0], 'r')
    # plt.plot(t_arr, bias_drift_error[:,1], 'g')
    # plt.plot(t_arr, bias_drift_error[:,2], 'b')
    # plt.subplot(2,1,2)
    # plt.title('Gyro bias error')
    # plt.plot(t_arr, bias_drift_error[:,0+3], 'r')
    # plt.plot(t_arr, bias_drift_error[:,1+3], 'g')
    # plt.plot(t_arr, bias_drift_error[:,2+3], 'b')
    # plt.savefig(FIG_DIR_PATH+'fac_bias_drift_error_{}.png'.format(i))
    # if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Factor Pose err'+str(i))
    for k in range(2):
        plt.subplot(2,1,1+k)
        plt.title('PO'[k])
        plt.plot(tkf_arr, fac_pose_err[:,0+2*k], 'r')
        plt.plot(tkf_arr, fac_pose_err[:,1+2*k], 'g')
        plt.plot(tkf_arr, fac_pose_err[:,2+2*k], 'b')
    plt.savefig(FIG_DIR_PATH+'fac_pose_errors_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)
    
    ############
    # PARAMETERS
    ############
    fig = plt.figure('Extrinsics MOCAP'+str(i))
    plt.title('Extrinsics MOCAP\n{}'.format(config))
    plt.subplot(2,1,1)
    plt.title('P')
    plt.plot(t_arr, extr_mocap_fbk[:,0], 'r')
    plt.plot(t_arr, extr_mocap_fbk[:,1], 'g')
    plt.plot(t_arr, extr_mocap_fbk[:,2], 'b')
    # plt.xlabel('t (s)')
    plt.ylabel('i_p_im (m)')
    plt.subplot(2,1,2)
    plt.title('O')
    plt.plot(t_arr, extr_mocap_fbk[:,3], 'r')
    plt.plot(t_arr, extr_mocap_fbk[:,4], 'g')
    plt.plot(t_arr, extr_mocap_fbk[:,5], 'b')    
    # plt.plot(t_arr, extr_mocap_fbk[:,6], 'k')  
    plt.ylabel('i_q_m (rad)')  
    plt.savefig(FIG_DIR_PATH+'extr_mocap_{}.png'.format(i))
    if CLOSE: plt.close(fig=fig)
    

    fig = plt.figure('IMU biases'+str(i))
    plt.subplot(2,1,1)
    plt.title('IMU biases\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, imu_bias[:,i],     'rgb'[i], label='est')
        plt.plot(t_arr, imu_bias_fbk[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(tkf_arr, imu_bias_fbk[-1,i]+envel_bi[:,i],   'rgb'[i]+'--', label='cov')
        plt.plot(tkf_arr, imu_bias_fbk[-1,i]-envel_bi[:,i],  'rgb'[i]+'--')
    plt.ylabel('bias acc (m/s^2)')
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(t_arr, imu_bias[:,3+i],     'rgb'[i], label='est')
        plt.plot(t_arr, imu_bias_fbk[:,3+i], 'rgb'[i]+'.', label='fbk')
        plt.plot(tkf_arr, imu_bias_fbk[-1,3+i]+envel_bi[:,3+i],  'rgb'[i]+'--', label='cov')
        plt.plot(tkf_arr, imu_bias_fbk[-1,3+i]-envel_bi[:,3+i],  'rgb'[i]+'--')
    plt.xlabel('t (s)')
    plt.ylabel('bias gyro (rad/s)')
    plt.legend()
    # plt.savefig(FIG_DIR_PATH+'imu_bias_{}.png'.format(i))
    plt.savefig(FIG_DIR_PATH+'imu_bias_{}.eps'.format(i), format='eps')
    if CLOSE: plt.close(fig=fig)

    





    #######
    # DRIFT: influence of windowed optim
    #######
    # fig = plt.figure()
    # plt.title('Position Diff\n{}'.format(config))
    # plt.plot(t_arr, o_p_ob_diff[:,0], 'r')
    # plt.plot(t_arr, o_p_ob_diff[:,1], 'g')
    # plt.plot(t_arr, o_p_ob_diff[:,2], 'b')
    # plt.xlabel('t (s)')
    # plt.ylabel('P (m)')
    # plt.savefig(FIG_DIR_PATH+'diff_pos_{}.png'.format(i))
    # if CLOSE: plt.close(fig=fig)

    # fig = plt.figure()
    # plt.title('Velocity Diff\n{}'.format(config))
    # plt.plot(t_arr, o_v_ob_diff[:,0], 'r')
    # plt.plot(t_arr, o_v_ob_diff[:,1], 'g')
    # plt.plot(t_arr, o_v_ob_diff[:,2], 'b')
    # plt.xlabel('t (s)')
    # plt.ylabel('V (m/s)')
    # plt.savefig(FIG_DIR_PATH+'diff_vel_{}.png'.format(i))
    # if CLOSE: plt.close(fig=fig)

if SHOW: plt.show()

plt.figure('POS ')
sns.heatmap(rmse_pos_arr[:,:,0].T, annot=True, linewidths=.5, xticklabels=max_t_kf_lst, yticklabels=KF_nb_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')

plt.figure('VEL')
sns.heatmap(rmse_vel_arr[:,:,0].T, annot=True, linewidths=.5, xticklabels=max_t_kf_lst, yticklabels=KF_nb_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')

traj_dur = t_arr[-1] - t_arr[0]
plt.figure('Compute time (% traj T)')
sns.heatmap(compute_time_arr[:,:,0].T/traj_dur, annot=True, linewidths=.5, xticklabels=max_t_kf_lst, yticklabels=KF_nb_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')


 
restore_initial_file(PARAM_FILE)
restore_initial_file(PROC_IMU_PARAM_FILE)
restore_initial_file(TREE_PARAM_FILE)


