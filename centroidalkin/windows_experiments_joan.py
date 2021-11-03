#!/usr/bin/env python
# coding: utf-8

"""
Goal: find a satisfying operational point for different trajectories
by varying the sliding window parameters
"""

import os
import sys
import time
import math
from pinocchio.deprecated import XYZQUATToSe3
import yaml
import shutil
import numpy as np
import subprocess
import itertools
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
import seaborn as sns
import pinocchio as pin


SHOW = True
RUN = True

# FILE_TYPE = 'jpg'
FILE_TYPE = 'png'
# FILE_TYPE = 'pdf'
CLOSE = not SHOW
COV = False

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

def restore_initial_file(path):
    path_bak = path+'_bak'
    # shutil.move(path_bak, path)
    shutil.copyfile(path_bak, path)

def diff_shift(arr):
    diff = np.roll(arr, -1, axis=0) - arr
    # last element/line will be huge since roll wraps around the array
    # -> just copy the n_last - 1
    diff[-1,:] = diff[-2,:]
    return diff

def qv2R(qvec):
    return pin.Quaternion(qvec.reshape((4,1))).toRotationMatrix()


# executable and param files paths
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
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/SinStamping_Corrected_09_12_2020/data_2020_12_09_17_54_format.npz'  # sin
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
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_59_format.npz'  # Already in air, random movements (15s), mocap 500Hz
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_16_03_format.npz'  # //



# New data set with solo in hand, mocap at 500Hz, some missing mocap data is interpolated using linear and slerp
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_2_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_3_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_4_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_3_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_4_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_5_format.npz'  # small jump


# New data to reintroduce Kinematics
# _calib: ~15 seconds of solo held in hand for a calibration procedure
# _move: ~10 seconds of solo on the ground realizing a move
# Air calibration
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_16_48_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_04_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_08_calib_format.npz'  ######
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_11_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_16_calib_format.npz'



# Ground movement
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_16_48_move_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_04_move_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_08_move_format.npz'  ######
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_11_move_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_16_move_format.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_19_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_19_move_format.npz'



######
# SIMU
######
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_gentle_backright_fflyer_format.npz'


# New data to reintroduce kinematics POINTFEET
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_36_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_38_calib_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_39_calib_format.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_36_move_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_38_move_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_39_move_format.npz'



######
# IRI 10/21
######
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_bis_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_in_air_25_45_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters_format.npz'


######
# LAAS 10/21
######
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_10s_format.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_1min_0_10_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_1min_23_34_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_1min_36_45_format.npz'


######
# IRI 10/21 SECOND TIME
######

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_15_24_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_24_34_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_34_44_format.npz'


# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_2min_6_42_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_2min_43_74_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_2min_74_117_format.npz'


# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_bis_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_bzzz_format.npz'


# WALKING TRAJ
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_format.npz'
params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_bis_format.npz'



# Choose problem statement according to the data file
RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_kine'
###### RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_kine_mocap'
# RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_mocap'
# RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_kine_mocap'

MOCAP_ALIGN = True  # align est traj wrt. mocap



traj_name = params['data_file_path'].split('/')[-1].split('.npz')[0]

# other main params
params['unfix_extr_sensor_pose'] = False


# Prior (for pbes without mocap)
params['std_prior_p'] = 0.001
params['std_prior_o'] = math.radians(90)
params['std_prior_v'] = 10



# MOCAP
params['std_pose_p'] = 0.001
params['std_pose_o_deg'] = 1
# i_pose_im = 6*[0] + [1]  # SIMU
# i_pose_im = [-0.1163, 0.0, -0.02,  0,0,0,1]  # NOMINAL, solo at laas
# i_pose_im = [0.11319228351109546, -0.025215545698249846, -0.01638201593610379, 
#             -0.00374592352960968, -6.326516868523575e-05, 0.004403909480908183, 0.9999832846781553]
i_pose_im = [-0.011159771171845226, -0.007321382374722619, -0.0024773021382709696, 
            0.0035047174131140808, 0.004237725032522593, 0.005168197514269616, 0.9999715237829807]  # imu mocap IRI good calib
i_pose_im = [-0.011159771171845226, -0.007321382374722619, -0.0024773021382709696, 
            0.0035047174131140808, 0.004237725032522593, 0.005168197514269616, 0.9999715237829807]  # imu mocap IRI good calib
i_pose_im = [-x for x in i_pose_im]      # TODO WHY ???    
params['i_p_im'] = i_pose_im[:3]
params['i_q_m'] = i_pose_im[3:]
params['std_mocap_extr_p'] = 0.01
params['std_mocap_extr_o_deg'] = 5

# IMU MOCAP extr
m_p_mb = np.array([-0.11872364, -0.0027602,  -0.01272801])
m_q_b = np.array([0.00698701, 0.00747303, 0.00597298, 0.99992983])
params['m_p_mb'] = m_p_mb.tolist() 
params['m_q_b'] = m_q_b.tolist()


m_T_b = pin.XYZQUATToSE3(np.concatenate([m_p_mb, m_q_b]))
b_T_m = m_T_b.inverse()
# i_T_m = pin.XYZQUATToSE3(i_pose_im)
# b_T_i = b_T_m*i_T_m.inverse()

# print('i_T_m:')
# print('xzy', i_T_m.translation)
# print('rpy', np.rad2deg(pin.rpy.matrixToRpy(i_T_m.rotation)))

# print('b_T_m:')
# print('xzy', b_T_m.translation)
# print('rpy', np.rad2deg(pin.rpy.matrixToRpy(b_T_m.rotation)))

# print('b_T_i:')
# print('xzy', b_T_i.translation)
# print('rpy', np.rad2deg(pin.rpy.matrixToRpy(b_T_i.rotation)))

# std kinematic factor
params['std_odom3d_est'] = 0.05  # m/(s^2 sqrt(Hz))


# IMU params
params_sensor_imu['motion_variances']['a_noise'] =       0.02    # standard deviation of Acceleration noise (same for all the axis) in m/s2
params_sensor_imu['motion_variances']['w_noise'] =       0.03    # standard deviation of Gyroscope noise (same for all the axis) in rad/sec
params_sensor_imu['motion_variances']['ab_rate_stdev'] = 1e-6    # m/s2/sqrt(s)           
params_sensor_imu['motion_variances']['wb_rate_stdev'] = 1e-6    # rad/s/sqrt(s)
# PRIOR IMU
params['bias_imu_prior'] = [0]*6
# params['std_abs_bias_acc'] =  1
# params['std_abs_bias_gyro'] = 1
# params['bias_imu_prior'] = [-0.008309301319705645, -0.003781564817763668, -0.00973822192542853, 0.005811674117089268, 0.00584121549049778, -0.0010103139949982967]  # calib traj
params['std_abs_bias_acc'] =  100
params['std_abs_bias_gyro'] = 100

params['dt'] = 1e-3  # 1 kHz
# params['dt'] = 2e-3  # 500 Hz
# params['max_t'] = 2.001
# params['max_t'] = 10.001
params['max_t'] = 100.001



# Load the input data (in case glitch with mocap...)

input_dic = np.load(params['data_file_path'], params['dt'])




###############################
###############################
###############################
###############################
std_pose_p = params['std_pose_p']
std_pose_o_deg = params['std_pose_o_deg']


# time_BAK_shift_mocap_lst = [-0.020,] # IRI 2nd optimal imu+mocap
i_p_im_lst = [
    [-0.01, -0.007, -0.0025],
    # [-0.02, -0.007, -0.0025],
    # [ 0.00, -0.007, -0.0025],
    ] # IRI 2nd optimal imu+mocap
# i_p_im_lst = [-0.020,] # IRI 2nd optimal imu+mocap
# std_odom3d_est_lst = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
# std_odom3d_est_lst = [0.001, 0.01, 0.3]
std_odom3d_est_lst = [0.01]





delta_qa_lst = [
    12*[0.0],
    # [ 0.0041, -0.0149, -0.0140, -0.0192, -0.0004, -0.0033,
    #   0.0064,  0.0046,  0.0212,  0.0097,  0.0234,  0.0190, ],
    # [-0.0041, 0.0149,    0.0140, 0.0192, 0.0004, 0.0033,
    #  -0.0064, -0.0046,  -0.0212,  +0.0097,  -0.0234,  -0.0190, ]
]

i_p_im_idx_lst = np.arange(len(i_p_im_lst))
std_odom3d_est_idx_lst = np.arange(len(std_odom3d_est_lst))
delta_qa_idx_lst = np.arange(len(delta_qa_lst))


###############################
###############################
###############################



possibs = nb_possibilities([i_p_im_lst, std_odom3d_est_lst, delta_qa_lst])
print('Combinations to evaluate: ', possibs)


RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out'

rmse_pos_arr = np.zeros((len(i_p_im_lst), len(std_odom3d_est_lst), len(delta_qa_lst)))
rmse_vel_arr = np.zeros((len(i_p_im_lst), len(std_odom3d_est_lst), len(delta_qa_lst)))
compute_time_arr = np.zeros((len(i_p_im_lst), len(std_odom3d_est_lst), len(delta_qa_lst)))


std_pose_p = params['std_pose_p']
std_pose_o_deg = params['std_pose_o_deg']


for idx_exp, (i_p_im_idx, std_odom3d_est_idx, delta_qa_idx) in enumerate(itertools.product(i_p_im_idx_lst, std_odom3d_est_idx_lst, delta_qa_idx_lst)):
    i_p_im = i_p_im_lst[i_p_im_idx]
    std_odom3d_est = std_odom3d_est_lst[std_odom3d_est_idx]
    delta_qa = delta_qa_lst[delta_qa_idx]


    params['out_npz_file_path'] = RESULTS+str(idx_exp)+'.npz'


    params['delta_qa'] = delta_qa
    params['std_odom3d_est'] = std_odom3d_est
    params['i_p_im'] = i_p_im

    # scale mocap
    # params['std_pose_p'] = scale_mocap*std_pose_p
    # params['std_pose_o_deg'] = scale_mocap*std_pose_o_deg

    with open(PARAM_FILE, 'w') as fw: yaml.dump(params, fw)

    with open(SENSOR_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_sensor_imu, fw)

    # params_proc_imu['keyframe_vote']['max_time_span'] = max_t_kf
    with open(PROC_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_proc_imu, fw)

    # params_tree['config']['problem']['tree_manager']['n_frames'] = KF_nb
    with open(TREE_PARAM_FILE, 'w') as fw: yaml.dump(params_tree, fw)



    # run executable
    t1 = time.time()
    # if RUN:  subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL)
    if RUN: subprocess.run(RUN_FILE)
    compute_time = time.time()-t1
    compute_time_arr[i_p_im_idx, std_odom3d_est_idx, delta_qa_idx] = compute_time 
    print(idx_exp, ':', compute_time)
    
    config = {
        'i_p_im': i_p_im,
        'std_odom3d_est': std_odom3d_est,
        'delta_qa': delta_qa
    }
    with open(FIG_DIR_PATH+'conf_{}.yaml'.format(idx_exp), 'w') as fw: yaml.dump(config, fw)

    # load raw results and compute rmses
    arr_dic = np.load(params['out_npz_file_path'])
    
    # Common data
    t_arr =          arr_dic['t']
    w_p_wm_gtr_arr = arr_dic['w_p_wm']
    w_q_m_gtr_arr =  arr_dic['w_q_m']
    w_v_wm_gtr_arr = arr_dic['w_v_wm']
    i_omg_oi_arr =   arr_dic['i_omg_oi']
    qa_arr =         arr_dic['qa']

    o_p_ob_arr = arr_dic['o_p_ob']
    o_q_b_arr =  arr_dic['o_q_b']
    o_v_ob_arr = arr_dic['o_v_ob']
    o_p_oi_arr = arr_dic['o_p_oi']
    o_q_i_arr =  arr_dic['o_q_i']
    o_v_oi_arr = arr_dic['o_v_oi']

    o_p_ob_fbk_arr = arr_dic['o_p_ob_fbk']
    o_q_b_fbk_arr =  arr_dic['o_q_b_fbk']
    o_v_ob_fbk_arr = arr_dic['o_v_ob_fbk']
    o_p_oi_fbk_arr = arr_dic['o_p_oi_fbk']
    o_q_i_fbk_arr =  arr_dic['o_q_i_fbk']
    o_v_oi_fbk_arr = arr_dic['o_v_oi_fbk']

    o_p_ob_diff = diff_shift(o_p_ob_arr)
    o_v_ob_diff = diff_shift(o_v_ob_arr)
    o_p_ob_fbk_diff = diff_shift(o_p_ob_fbk_arr)
    o_v_ob_fbk_diff = diff_shift(o_v_ob_fbk_arr)
    # rmse_pos_arr[i_p_im_idx, std_odom3d_est_idx, delta_qa_idx] = rmse(o_p_ob_diff).mean()
    # rmse_vel_arr[i_p_im_idx, std_odom3d_est_idx, delta_qa_idx] = rmse(o_v_ob_diff).mean()
    

    # biases and extrinsics
    imu_bias = arr_dic['imu_bias']
    imu_bias_fbk = arr_dic['imu_bias_fbk']
    extr_mocap_fbk = arr_dic['i_pose_im_fbk']
    i_T_m = pin.XYZQUATToSE3(extr_mocap_fbk[-1,:])

    # SE3
    o_T_i_lst = [pin.XYZQUATToSE3(np.concatenate([o_p_oi, o_q_i])) for o_p_oi, o_q_i in zip(o_p_oi_arr, o_q_i_arr)]
    o_T_i_fbk_lst = [pin.XYZQUATToSE3(np.concatenate([o_p_oi, o_q_i])) for o_p_oi, o_q_i in zip(o_p_oi_fbk_arr, o_q_i_fbk_arr)]
    w_T_m_gtr_lst = [pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m])) for w_p_wm, w_q_m in zip(w_p_wm_gtr_arr, w_q_m_gtr_arr)]




    ##########################################
    ##########################################
    # ALIGN TRAJECTORY WITH GLOBAL MOCAP FRAME
    ##########################################
    ##########################################


    w_T_m_init = w_T_m_gtr_lst[0] if MOCAP_ALIGN else pin.SE3.Identity()
    o_T_i_init = o_T_i_lst[0] if MOCAP_ALIGN else pin.SE3.Identity()
    o_T_m_init = o_T_i_init*i_T_m  # align trajectories wrt. local mocap frame

    # transform estimated trajectories in mocap GLOBAL frame
    w_T_o = w_T_m_init * o_T_m_init.inverse()

    # w_T_m = w_T_o*o_T_i*i_T_m -> quantity to compare with direct mocap measurements  
    w_T_m_lst =     [w_T_o*o_T_i*i_T_m for o_T_i in o_T_i_lst]
    w_T_m_fbk_lst = [w_T_o*o_T_i*i_T_m for o_T_i in o_T_i_fbk_lst]

    # TODO: velocities...
    w_v_wm_arr = np.array([w_T_o.rotation@o_v_ob for o_v_ob in o_v_ob_arr])
    w_v_wm_fbk_arr = np.array([w_T_o.rotation@o_v_ob for o_v_ob in o_v_ob_fbk_arr])


    ###################################################
    # extract positions
    w_p_wm_arr =     np.array([w_T_m.translation for w_T_m in w_T_m_lst]) 
    w_p_wm_fbk_arr = np.array([w_T_m.translation for w_T_m in w_T_m_fbk_lst]) 
    w_p_wm_gtr_arr = np.array([w_T_m.translation for w_T_m in w_T_m_gtr_lst]) 

    # quaternion as angle axis in world frame
    w_rpy_m_arr =     np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_lst]) 
    w_rpy_m_fbk_arr = np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_fbk_lst]) 
    w_rpy_m_gtr_arr = np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_gtr_lst]) 

    # Compute velocities in base frame rather than global frame
    m_v_wm_arr =     np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_lst, w_v_wm_arr)])
    m_v_wm_fbk_arr = np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_fbk_lst, w_v_wm_fbk_arr)])
    m_v_wm_gtr_arr = np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_gtr_lst, w_v_wm_gtr_arr)])

    # Compute velocities due to lever arm and rotation
    m_p_mi = i_T_m.inverse().translation
    i_v_lever_arr = np.array([ np.cross(i_omg_oi-bi, i_T_m.rotation@m_p_mi) for i_omg_oi, bi  in zip(i_omg_oi_arr, imu_bias[:,3:])]) 

    # covariances
    Nsig = 2
    tkf_arr = arr_dic['tkf']
    Nkf = len(tkf_arr)
    # Qp = arr_dic['Qp']
    # Qo = arr_dic['Qo']
    # Qv = arr_dic['Qv']
    # Qbi = arr_dic['Qbi']
    # Qmp = arr_dic['Qm']
    Qp = arr_dic['Qp_fbk']
    Qo = arr_dic['Qo_fbk']
    Qv = arr_dic['Qv_fbk']
    Qbi = arr_dic['Qbi_fbk']
    Qm = arr_dic['Qm_fbk']
    envel_p =  Nsig*np.sqrt(Qp)
    envel_o =  Nsig*np.sqrt(Qo)
    envel_v =  Nsig*np.sqrt(Qv)
    envel_bi = Nsig*np.sqrt(Qbi)
    envel_m =  Nsig*np.sqrt(Qm)

    # bias cov can explode at the beginning
    envel_p = np.clip(envel_p, 3*[0], 3*[20])
    envel_v = np.clip(envel_v, 3*[0], 3*[1])
    envel_v = np.clip(envel_v, 3*[0], 3*[2])
    envel_bi = np.clip(envel_bi, 6*[0], 3*[1.5] + 3*[0.3])
    envel_m = np.clip(envel_m, 6*[0], 3*[1] + 3*[0.5])

    # factor errors
    fac_imu_err = arr_dic['fac_imu_err']
    fac_pose_err = arr_dic['fac_pose_err']

    i_pose_im_end = extr_mocap_fbk[-1,:]
    i_T_b = pin.XYZQUATToSE3(i_pose_im_end)
    b_T_i = i_T_b.inverse()
    i_pose_im = pin.SE3ToXYZQUAT(b_T_i)


    print(traj_name)
    np.set_printoptions(precision=10,
                        suppress=True,
                        linewidth=sys.maxsize,
                        sign='+',
                        floatmode='fixed')
    
    def commas(arr):
        return ', '.join(str(a) for a in arr)

    print('imu_bias END')
    print('val', commas(imu_bias[-1,:]))
    print('sig', commas(envel_bi[-1,:]))
    print('Mocap i_pose_im END')
    print('val', commas(i_pose_im))
    print('sig', commas(envel_m[-1,:]))
    # print('origin quaternion END')
    # print('val', o_q_b_arr[0,:])
    # print('sig', o_q_b_arr[0,:])




    #########################
    # Investigate Kine factor
    #########################
    from example_robot_data import load
    robot = load('solo12')
    N = len(qa_arr)
    ee_names = ["FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE"]
    ee_ids = [robot.model.getFrameId(ee_name) for ee_name in ee_names]

    err_lst = []
    N_btw_kf = 200
    for idx_kf in range(N_btw_kf, N, N_btw_kf):
        idx_kf_prev = idx_kf - N_btw_kf
        # q_prev = np.concatenate([o_p_ob_arr[idx_kf_prev,:], o_q_b_arr[idx_kf_prev,:], qa_arr[idx_kf_prev]])
        q_prev = np.concatenate([o_p_ob_arr[0,:], o_q_b_arr[0,:], qa_arr[0]])
        q = np.concatenate([o_p_ob_arr[idx_kf,:], o_q_b_arr[idx_kf,:], qa_arr[idx_kf]])
        o_p_ol_prev = robot.framePlacement(q_prev, ee_ids[0]).translation
        o_p_ol = robot.framePlacement(q, ee_ids[0]).translation
        err = o_p_ol_prev - o_p_ol
        err_lst.append(err)

    err_kin_arr = np.array(err_lst)

    fig = plt.figure('Kin factor error pinocchio'+str(idx_exp))
    plt.title('Kin factor error pinocchio\n{}'.format(config))
    for i in range(3):
        plt.plot(tkf_arr[:-1], err_kin_arr[:,i], 'rgb'[i], label='est')
    plt.ylim(-0.01, 0.01)
    plt.xlabel('t (s)')
    plt.ylabel('P (m)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'err_kin_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)


    #########################
    

    #######################
    # TRAJECTORY EST VS GTR
    #######################
    fig = plt.figure('Position est vs mocap'+str(i))
    plt.title('Position est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_p_wm_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_p_wm_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk', markersize=1)
        plt.plot(t_arr, w_p_wm_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('P (m)')
    # plt.ylim(0.67, 0.75)
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'pos_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Velocity est vs mocap GLOBAL frame'+str(i))
    plt.title('Velocity est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_v_wm_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_v_wm_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, w_v_wm_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Velocity est vs mocap BASE frame'+str(i))
    plt.title('Velocity est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, m_v_wm_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, m_v_wm_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, m_v_wm_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Velocity lever arm'+str(i))
    plt.title('Velocity lever arm\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, i_v_lever_arr[:,i], 'rgb'[i], label='est')
    plt.xlabel('t (s)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)


    fig = plt.figure('Orientation est vs mocap'+str(i))
    plt.title('Orientation est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_rpy_m_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_rpy_m_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, w_rpy_m_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('Q')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'vel_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)



    #############
    # ERROR plots
    #############
    title = 'Position error and covariances'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k, label='')
        plt.plot(t_arr, w_p_wm_arr[:,k] - w_p_wm_gtr_arr[:,k], 'm', label='est')
        plt.plot(t_arr, w_p_wm_fbk_arr[:,k] - w_p_wm_gtr_arr[:,k], 'c.', label='fbk')
        if COV: 
            plt.plot(tkf_arr,  envel_p[:,k], 'k', label='cov')
            plt.plot(tkf_arr, -envel_p[:,k], 'k', label='cov')
        plt.legend()
    plt.savefig(FIG_DIR_PATH+'err_pos_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)
    
    title = 'Velocity error and covariances'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.plot(t_arr, w_v_wm_arr[:,k] - w_v_wm_gtr_arr[:,k], 'm', label='est')
        plt.plot(t_arr, w_v_wm_fbk_arr[:,k] - w_v_wm_gtr_arr[:,k], 'c.', label='fbk')
        if COV: 
            plt.plot(tkf_arr,  envel_v[:,k], 'k', label='cov')
            plt.plot(tkf_arr, -envel_v[:,k], 'k', label='cov')
        plt.legend()
    plt.savefig(FIG_DIR_PATH+'err_vel_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    err_o =     np.array([pin.log(w_T_m.rotation*w_T_m_gtr.rotation.T) for w_T_m, w_T_m_gtr in zip(w_T_m_lst, w_T_m_gtr_lst)])
    err_o_fbk = np.array([pin.log(w_T_m.rotation*w_T_m_gtr.rotation.T) for w_T_m, w_T_m_gtr in zip(w_T_m_fbk_lst, w_T_m_gtr_lst)])

    title = 'Orientation error and covariances'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.plot(t_arr, err_o[:,k], 'm', label='est')
        plt.plot(t_arr, err_o_fbk[:,k], 'c.', label='fbk')
        if COV: 
            plt.plot(tkf_arr,  envel_o[:,k], 'k', label='cov')
            plt.plot(tkf_arr, -envel_o[:,k], 'k', label='cov')
        plt.legend()
    plt.savefig(FIG_DIR_PATH+'err_rot_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    ###############
    # FACTOR ERRORS
    ###############
    title = 'Factor IMU err'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))    
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.title('POV'[k])
        plt.plot(tkf_arr, fac_imu_err[:,0+3*k], 'r')
        plt.plot(tkf_arr, fac_imu_err[:,1+3*k], 'g')
        plt.plot(tkf_arr, fac_imu_err[:,2+3*k], 'b')
    plt.savefig(FIG_DIR_PATH+'fac_imu_errors_{}.{}'.format(idx_exp, FILE_TYPE))
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
    # plt.savefig(FIG_DIR_PATH+'fac_bias_drift_res_{}.{}'.format(idx_exp, FILE_TYPE))
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
    # plt.savefig(FIG_DIR_PATH+'fac_bias_drift_error_{}.{}'.format(idx_exp, FILE_TYPE))
    # if CLOSE: plt.close(fig=fig)

    title = 'Factor Pose err'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))    
    for k in range(2):
        plt.subplot(2,1,1+k)
        plt.title('PO'[k])
        plt.plot(tkf_arr, fac_pose_err[:,0+2*k], 'r')
        plt.plot(tkf_arr, fac_pose_err[:,1+2*k], 'g')
        plt.plot(tkf_arr, fac_pose_err[:,2+2*k], 'b')
    plt.savefig(FIG_DIR_PATH+'fac_pose_errors_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)
    
    ############
    # PARAMETERS
    ############
    title = 'Extrinsics MOCAP'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))    
    plt.subplot(2,1,1)
    plt.title('P')
    for i in range(3):
        plt.plot(t_arr, extr_mocap_fbk[:,i], 'rgb'[i])
        if COV: 
            plt.plot(tkf_arr, extr_mocap_fbk[-1,i]+envel_m[:,i], 'rgb'[i]+'--', label='cov')
            plt.plot(tkf_arr, extr_mocap_fbk[-1,i]-envel_m[:,i], 'rgb'[i]+'--')
    # plt.xlabel('t (s)')
    plt.ylabel('i_p_im (m)')
    plt.subplot(2,1,2)
    plt.title('O')
    for i in range(3):
        plt.plot(t_arr, extr_mocap_fbk[:,3+i], 'rgb'[i])
        if COV: 
            plt.plot(tkf_arr, extr_mocap_fbk[-1,3+i]+envel_m[:,3+i], 'rgb'[i]+'--', label='cov')
            plt.plot(tkf_arr, extr_mocap_fbk[-1,3+i]-envel_m[:,3+i], 'rgb'[i]+'--')
    # plt.plot(t_arr, extr_mocap_fbk[:,6], 'k')  
    plt.ylabel('i_q_m (rad)')  
    plt.savefig(FIG_DIR_PATH+'extr_mocap_{}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)
    
    title = 'IMU biases'
    fig = plt.figure(title+str(i))
    plt.suptitle(title+'\n{}'.format(config))    
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(t_arr, imu_bias[:,i],     'rgb'[i], label='est')
        plt.plot(t_arr, imu_bias_fbk[:,i], 'rgb'[i]+'.', label='fbk')
        if COV: 
            plt.plot(tkf_arr, imu_bias_fbk[-1,i]+envel_bi[:,i],   'rgb'[i]+'--', label='cov')
            plt.plot(tkf_arr, imu_bias_fbk[-1,i]-envel_bi[:,i],  'rgb'[i]+'--')
    plt.ylabel('bias acc (m/s^2)')
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(t_arr, imu_bias[:,3+i],     'rgb'[i], label='est')
        plt.plot(t_arr, imu_bias_fbk[:,3+i], 'rgb'[i]+'.', label='fbk')
        if COV: 
            plt.plot(tkf_arr, imu_bias_fbk[-1,3+i]+envel_bi[:,3+i],  'rgb'[i]+'--', label='cov')
            plt.plot(tkf_arr, imu_bias_fbk[-1,3+i]-envel_bi[:,3+i],  'rgb'[i]+'--')
    plt.xlabel('t (s)')
    plt.ylabel('bias gyro (rad/s)')
    plt.legend()
    plt.savefig(FIG_DIR_PATH+'imu_bias_{}.{}'.format(idx_exp, FILE_TYPE))
    # plt.savefig(FIG_DIR_PATH+'imu_bias_{}.eps'.format(idx_exp, FILE_TYPE), format='eps')
    if CLOSE: plt.close(fig=fig)

    

    #######
    # EXTRA
    #######
    omg_norm = np.linalg.norm(arr_dic['i_omg_oi'], axis=1)

    fig = plt.figure('P jumps = f(omg)')
    plt.title('P jumps=f(gyro)'+'\n{}'.format(config))
    for i in range(3):
        plt.plot(omg_norm, o_p_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    plt.savefig(FIG_DIR_PATH+'jump_P_fbk_f(gyro){}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('P jumps = f(t)')
    plt.title('P jumps=f(t)'+'\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, o_p_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    plt.savefig(FIG_DIR_PATH+'jump_P_fbk_f(t){}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)
    
    fig = plt.figure('V jumps = f(gyro)')
    plt.title('v jumps=f(gyro)'+'\n{}'.format(config))
    for i in range(3):
        plt.plot(omg_norm, o_v_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    plt.savefig(FIG_DIR_PATH+'jump_V_fbk_f(gyro){}.{}'.format(idx_exp, FILE_TYPE))
    if CLOSE: plt.close(fig=fig)


if SHOW: plt.show()

plt.figure('POS ')
sns.heatmap(rmse_pos_arr[:,:,0].T, annot=True, linewidths=.5, xticklabels=i_p_im_lst, yticklabels=std_odom3d_est_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')

plt.figure('VEL')
sns.heatmap(rmse_vel_arr[:,:,0].T, annot=True, linewidths=.5, xticklabels=i_p_im_lst, yticklabels=std_odom3d_est_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')

traj_dur = t_arr[-1] - t_arr[0]
plt.figure('Compute time (% traj T)')
sns.heatmap(compute_time_arr[:,:,0].T/traj_dur, annot=True, linewidths=.5, xticklabels=i_p_im_lst, yticklabels=std_odom3d_est_lst)
plt.xlabel('max_t_kf')
plt.ylabel('KF_nb')


 
restore_initial_file(PARAM_FILE)
restore_initial_file(PROC_IMU_PARAM_FILE)
restore_initial_file(TREE_PARAM_FILE)


