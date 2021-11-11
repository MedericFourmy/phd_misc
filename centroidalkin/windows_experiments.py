#!/usr/bin/env python3
# coding: utf-8


"""
Notations follow (mostly): http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly

Frames:
o = global frame of the wolf estimate (defined by the first frame often)
w = global frame of the mocap
ω = global frame of the complementary filter
i = imu
b = base (root of kinematics)
m = mocap frame attached to the robot which pose is given in w  

Examples and different estimates provenances
o_p_oi = wolf a posterio estimate of the position of the imu  
o_p_oi_fbk = wolf online estimate of the position of the imu
w_p_wm_gtr = mocap position measurement
ω_T_b = pose esitmated by the complementary filter running on solo
"""


import os
import sys
import time
import math
from pinocchio.deprecated import jointJacobian
import yaml
import shutil
import numpy as np
import subprocess
import itertools
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
import pinocchio as pin


SHOW = True
RUN = True

FILE_TYPES = ['jpg', 'eps']
CLOSE = not SHOW
COV = False
BODYDYNAMICS_PATH = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics'

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

def save_figs(fig, name, format_list):
    for format in format_list:
        fig.savefig(name+'.{}'.format(format), format=format)



# executable and param files paths
PARAM_FILE = os.path.join(BODYDYNAMICS_PATH, 'demos/solo_real_estimation.yaml')
SENSOR_IMU_PARAM_FILE = os.path.join(BODYDYNAMICS_PATH, 'demos/sensor_imu_solo12.yaml')
PROC_IMU_PARAM_FILE = os.path.join(BODYDYNAMICS_PATH, 'demos/processor_imu_solo12.yaml')
TREE_PARAM_FILE = os.path.join(BODYDYNAMICS_PATH, 'demos/tree_manager.yaml')


params = create_bak_file_and_get_params(PARAM_FILE)
params_proc_imu = create_bak_file_and_get_params(PROC_IMU_PARAM_FILE)
params_sensor_imu = create_bak_file_and_get_params(SENSOR_IMU_PARAM_FILE)
params_tree = create_bak_file_and_get_params(TREE_PARAM_FILE)




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


# WALKING TRAJ (red -> reduced planned contact meas)
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_red_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_bis_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_bis_red_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_gait_10_10_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_gait_5_15_format.npz'

# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_alternate_walk_format.npz'




# LAAS 11/9: solo walking and trotting, with and without round feet

params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_trot_round_feet_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_walk_round_feet_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_trot_round_feet_with_yaw_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_trot_point_feet_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_trot_point_feet_with_yaw_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_walk_point_feet_format.npz'
# params['data_file_path'] = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_21_11_09/solo_trot_point_feet_with_yaw_format.npz'


# Choose problem demo to run
RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_kine'
###### RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_kine_mocap'
# RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_mocap'
# RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_imu_kine_mocap'


traj_name = params['data_file_path'].split('/')[-1].split('.npz')[0]
# add a lil' extra something for each expe
# traj_name += '_With_alt005'
traj_name += '_test'

FIG_DIR_PATH = 'figs/window_experiments/'+traj_name+'/'

shutil.rmtree(FIG_DIR_PATH, ignore_errors=True)

if not os.path.exists(FIG_DIR_PATH):
    print('Create '+FIG_DIR_PATH+' folder')
    os.makedirs(FIG_DIR_PATH)


# CERES Solver
params['max_num_iterations'] = 1000
params['func_tol'] = 1e-8
params['compute_cov'] = False


# other main params
params['unfix_extr_sensor_pose'] = False


# Prior (for demos without mocap like solo_imu_kine)
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
i_pose_im = [-0.01116, -0.00732, -0.00248, 
            0.00350, 0.00424, 0.0052, 0.9999]  # imu mocap IRI good calib
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

# std kinematic factor
params['std_foot_nomove'] = 0.05  # m/(s^2 sqrt(Hz))
params['std_altitude'] = 0.05  # m/(s^2 sqrt(Hz))


# IMU params
params_sensor_imu['motion_variances']['a_noise'] =       0.02    # standard deviation of Acceleration noise (same for all the axis) in m/s2
params_sensor_imu['motion_variances']['w_noise'] =       0.03    # standard deviation of Gyroscope noise (same for all the axis) in rad/sec
params_sensor_imu['motion_variances']['ab_rate_stdev'] = 1e-6    # m/s2/sqrt(s)           
params_sensor_imu['motion_variances']['wb_rate_stdev'] = 1e-6    # rad/s/sqrt(s)
# PRIOR IMU
params['bias_imu_prior'] = [0]*6
# params['std_abs_bias_acc'] =  1
# params['std_abs_bias_gyro'] = 1
params['std_abs_bias_acc'] =  100
params['std_abs_bias_gyro'] = 100

params['dt'] = 1e-3  # 1 kHz
# params['max_t'] = 1.0
# params['max_t'] = 20.001
params['max_t'] = 60.001
# params['max_t'] = 90.001


# KF management
params_proc_imu['keyframe_vote']['max_time_span'] = 0.05
# params_proc_imu['keyframe_vote']['max_time_span'] = 0.1
# params_proc_imu['keyframe_vote']['max_time_span'] = 0.2

params_tree['config']['problem']['tree_manager']['n_frames'] = 5000000000


# Parameters found by Nicolas on last calibration
alpha_qa_gold = [-0.12742363, -0.00529146, -0.03104106, -0.14280791, -0.02257373,
                 -0.04482504, -0.08464646, -0.10036708,  0.00414789, -0.07920959, -0.01540801, -0.04312102]
delta_qa_gold = [ 0.00397903, -0.00273951,  0.00580113,  0.02760998,  0.01361366,
                     -0.00809446, -0.00788157, -0.00411362,  0.01018828,  0.01604624, -0.00141011, -0.01212298]

params['delta_qa'] = delta_qa_gold
params['alpha_qa'] = alpha_qa_gold




###############################
# Parameters on which we wish to loop
# If 
###############################
###############################

# time_BAK_shift_mocap_lst = [-0.020,] # IRI 2nd optimal imu+mocap
alpha_qa_lst = [
    # 12*[0.0],
    alpha_qa_gold,
    ] # IRI 2nd optimal imu+mocap

# n_frames_lst = [5, 10, 100, 1000]
n_frames_lst = [50000]

delta_qa_lst = [
    12*[0.0],
    # delta_qa_gold,
]

alpha_qa_idx_lst = np.arange(len(alpha_qa_lst))
n_frames_idx_lst = np.arange(len(n_frames_lst))
delta_qa_idx_lst = np.arange(len(delta_qa_lst))


###############################
###############################
###############################



possibs = nb_possibilities([alpha_qa_lst, n_frames_lst, delta_qa_lst])
print('Combinations to evaluate: ', possibs)


RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out'

rmse_pos_arr = np.zeros((len(alpha_qa_lst), len(n_frames_lst), len(delta_qa_lst)))
rmse_vel_arr = np.zeros((len(alpha_qa_lst), len(n_frames_lst), len(delta_qa_lst)))
compute_time_arr = np.zeros((len(alpha_qa_lst), len(n_frames_lst), len(delta_qa_lst)))


std_pose_p = params['std_pose_p']
std_pose_o_deg = params['std_pose_o_deg']



# Recover some stuff from the original data file
# traj_name += '_With_alt005'

# sys.path.append(os.path.join(os.getcwd(), '../quadruest') )
# from data_readers import shortened_arr_dic


data_dic = np.load(params['data_file_path'])
ω_T_b_lst = [pin.XYZQUATToSE3(ω_xyzquat_b) for ω_xyzquat_b in data_dic['esti_q_filt'][:,:7]]
b_v_ωb_cpf_arr = data_dic['esti_filt_lin_vel']


for idx_exp, (alpha_qa_idx, n_frames_idx, delta_qa_idx) in enumerate(itertools.product(alpha_qa_idx_lst, n_frames_idx_lst, delta_qa_idx_lst)):
    alpha_qa = alpha_qa_lst[alpha_qa_idx]
    n_frames = n_frames_lst[n_frames_idx]
    delta_qa = delta_qa_lst[delta_qa_idx]


    params['out_npz_file_path'] = RESULTS+str(idx_exp)+'.npz'


    params['delta_qa'] = delta_qa
    # params['n_frames'] = n_frames
    params['alpha_qa'] = alpha_qa

    # scale mocap
    # params['std_pose_p'] = scale_mocap*std_pose_p
    # params['std_pose_o_deg'] = scale_mocap*std_pose_o_deg

    with open(PARAM_FILE, 'w') as fw: yaml.dump(params, fw)

    with open(SENSOR_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_sensor_imu, fw)

    # params_proc_imu['keyframe_vote']['max_time_span'] = max_t_kf
    with open(PROC_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_proc_imu, fw)

    params_tree['config']['problem']['tree_manager']['n_frames'] = n_frames
    with open(TREE_PARAM_FILE, 'w') as fw: yaml.dump(params_tree, fw)



    # run executable
    t1 = time.time()
    # if RUN:  subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL)
    if RUN: subprocess.run(RUN_FILE)
    compute_time = time.time()-t1
    compute_time_arr[alpha_qa_idx, n_frames_idx, delta_qa_idx] = compute_time 
    print(idx_exp, ':', compute_time)
    
    config = {
        'alpha_qa': alpha_qa,
        'n_frames': n_frames,
        'delta_qa': delta_qa
    }
    with open(FIG_DIR_PATH+'conf_{}.yaml'.format(idx_exp), 'w') as fw: yaml.dump(config, fw)

    # load raw results and compute rmses
    res_dic = np.load(params['out_npz_file_path'])
    t_arr =          res_dic['t']

    # shorten stuff from original dataset
    ω_T_b_lst = ω_T_b_lst[:t_arr.shape[0]]
    b_v_ωb_cpf_arr = b_v_ωb_cpf_arr[:t_arr.shape[0],:]

    # Common data
    w_p_wm_gtr_arr = res_dic['w_p_wm']
    w_q_m_gtr_arr =  res_dic['w_q_m']
    w_v_wm_gtr_arr = res_dic['w_v_wm']
    i_omg_oi_arr =   res_dic['i_omg_oi']
    qa_arr =         res_dic['qa']

    o_p_ob_arr = res_dic['o_p_ob']
    o_q_b_arr =  res_dic['o_q_b']
    o_v_ob_arr = res_dic['o_v_ob']
    o_p_oi_arr = res_dic['o_p_oi']
    o_q_i_arr =  res_dic['o_q_i']
    o_v_oi_arr = res_dic['o_v_oi']

    o_p_ob_fbk_arr = res_dic['o_p_ob_fbk']
    o_q_b_fbk_arr =  res_dic['o_q_b_fbk']
    o_v_ob_fbk_arr = res_dic['o_v_ob_fbk']
    o_p_oi_fbk_arr = res_dic['o_p_oi_fbk']
    o_q_i_fbk_arr =  res_dic['o_q_i_fbk']
    o_v_oi_fbk_arr = res_dic['o_v_oi_fbk']

    o_p_ob_diff = diff_shift(o_p_ob_arr)
    o_v_ob_diff = diff_shift(o_v_ob_arr)
    o_p_ob_fbk_diff = diff_shift(o_p_ob_fbk_arr)
    o_v_ob_fbk_diff = diff_shift(o_v_ob_fbk_arr)
    # rmse_pos_arr[alpha_qa_idx, n_frames_idx, delta_qa_idx] = rmse(o_p_ob_diff).mean()
    # rmse_vel_arr[alpha_qa_idx, n_frames_idx, delta_qa_idx] = rmse(o_v_ob_diff).mean()
    

    # biases and extrinsics
    imu_bias = res_dic['imu_bias']
    imu_bias_fbk = res_dic['imu_bias_fbk']
    extr_mocap_fbk = res_dic['i_pose_im_fbk']
    i_T_m = pin.XYZQUATToSE3(extr_mocap_fbk[-1,:])

    i_pose_im_end = extr_mocap_fbk[-1,:]
    i_T_b = pin.XYZQUATToSE3(i_pose_im_end)
    b_T_i = i_T_b.inverse()
    i_pose_im = pin.SE3ToXYZQUAT(b_T_i)

    # SE3
    o_T_i_lst = [pin.XYZQUATToSE3(np.concatenate([o_p_oi, o_q_i])) for o_p_oi, o_q_i in zip(o_p_oi_arr, o_q_i_arr)]
    o_T_i_fbk_lst = [pin.XYZQUATToSE3(np.concatenate([o_p_oi, o_q_i])) for o_p_oi, o_q_i in zip(o_p_oi_fbk_arr, o_q_i_fbk_arr)]
    w_T_m_gtr_lst = [pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m])) for w_p_wm, w_q_m in zip(w_p_wm_gtr_arr, w_q_m_gtr_arr)]




    ##########################################
    ##########################################
    # ALIGN TRAJECTORY WITH GLOBAL MOCAP FRAME
    ##########################################
    ##########################################

    w_T_m_init = w_T_m_gtr_lst[0]
    o_T_m_init = o_T_i_lst[0]*i_T_m
    ω_T_m_init = ω_T_b_lst[0]*b_T_m

    # transform estimated trajectories in mocap GLOBAL frame
    w_T_o = w_T_m_init * o_T_m_init.inverse()
    w_T_ω = w_T_m_init * ω_T_m_init.inverse()

    # w_T_m = w_T_o*o_T_i*i_T_m -> quantity to compare with direct mocap measurements  
    w_T_m_lst =     [w_T_o*o_T_i*i_T_m for o_T_i in o_T_i_lst]
    w_T_m_fbk_lst = [w_T_o*o_T_i*i_T_m for o_T_i in o_T_i_fbk_lst]
    w_T_m_cpf_lst = [w_T_ω*ω_T_b*b_T_m for ω_T_b in ω_T_b_lst]



    ###################################################
    # extract positions
    w_p_wm_arr =     np.array([w_T_m.translation for w_T_m in w_T_m_lst]) 
    w_p_wm_fbk_arr = np.array([w_T_m.translation for w_T_m in w_T_m_fbk_lst]) 
    w_p_wm_gtr_arr = np.array([w_T_m.translation for w_T_m in w_T_m_gtr_lst]) 
    w_p_wm_cpf_arr = np.array([w_T_m.translation for w_T_m in w_T_m_cpf_lst]) 

    # quaternion as angle axis in world frame
    w_rpy_m_arr =     np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_lst]) 
    w_rpy_m_fbk_arr = np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_fbk_lst]) 
    w_rpy_m_gtr_arr = np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_gtr_lst]) 
    w_rpy_m_cpf_arr = np.array([pin.rpy.matrixToRpy(w_T_m.rotation) for w_T_m in w_T_m_cpf_lst]) 

    # # Compute velocities in base frame rather than global frame
    # m_v_wm_arr =     np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_lst, w_v_wm_arr)])
    # m_v_wm_fbk_arr = np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_fbk_lst, w_v_wm_fbk_arr)])
    # m_v_wm_gtr_arr = np.array([w_T_m.rotation.T@w_v_wm for w_T_m, w_v_wm in zip(w_T_m_gtr_lst, w_v_wm_gtr_arr)])

    # TODO: velocities -> add omega x lever term
    # w_v_wm_arr = np.array([w_T_o.rotation@o_v_ob for o_v_ob in o_v_ob_arr])
    # w_v_wm_fbk_arr = np.array([w_T_o.rotation@o_v_ob for o_v_ob in o_v_ob_fbk_arr])

    # Compute base velocities in base frame rather than global frame
    o_R_b_lst = [o_T_i.rotation@i_T_b.rotation for o_T_i in o_T_i_lst]
    o_R_b_fbk_lst = [o_T_i.rotation@i_T_b.rotation for o_T_i in o_T_i_fbk_lst]
    b_v_ob_arr = np.array([o_R_b.T@o_v_ob for o_R_b, o_v_ob in zip(o_R_b_lst, o_v_ob_arr)])
    b_v_ob_fbk_arr = np.array([o_R_b.T@o_v_ob for o_R_b, o_v_ob in zip(o_R_b_fbk_lst, o_v_ob_fbk_arr)])

    # # Compute velocities due to lever arm and rotation (just for plotting...)
    # m_p_mi = i_T_m.inverse().translation
    # i_v_lever_arr = np.array([ np.cross(i_omg_oi-bi, i_T_m.rotation@m_p_mi) for i_omg_oi, bi  in zip(i_omg_oi_arr, imu_bias[:,3:])]) 

    # Covariances
    Nsig = 2
    tkf_arr = res_dic['tkf']
    Nkf = len(tkf_arr)
    # Qp = res_dic['Qp']
    # Qo = res_dic['Qo']
    # Qv = res_dic['Qv']
    # Qbi = res_dic['Qbi']
    # Qmp = res_dic['Qm']
    Qp = res_dic['Qp_fbk']
    Qo = res_dic['Qo_fbk']
    Qv = res_dic['Qv_fbk']
    Qbi = res_dic['Qbi_fbk']
    Qm = res_dic['Qm_fbk']
    envel_p =  Nsig*np.sqrt(Qp)
    envel_o =  Nsig*np.sqrt(Qo)
    envel_v =  Nsig*np.sqrt(Qv)
    envel_bi = Nsig*np.sqrt(Qbi)
    envel_m =  Nsig*np.sqrt(Qm)


    # bias cov can explode at the beginning -> clip them
    envel_p = np.clip(envel_p, 3*[0], 3*[20])
    envel_v = np.clip(envel_v, 3*[0], 3*[1])
    envel_v = np.clip(envel_v, 3*[0], 3*[2])
    envel_bi = np.clip(envel_bi, 6*[0], 3*[1.5] + 3*[0.3])
    envel_m = np.clip(envel_m, 6*[0], 3*[1] + 3*[0.5])

    # factor errors
    fac_imu_err = res_dic['fac_imu_err']
    fac_pose_err = res_dic['fac_pose_err']

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





    config = '.\n'.join(["{}: {}".format(k, v) for k, v in config.items()])
    

    #######################
    # TRAJECTORY EST VS GTR
    #######################
    fig = plt.figure('Position est vs mocap'+str(idx_exp))
    plt.title('Position est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_p_wm_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_p_wm_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk', alpha=0.5, markersize=1)
        plt.plot(t_arr, w_p_wm_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
        plt.plot(t_arr, w_p_wm_cpf_arr[:,i], 'rgb'[i]+'o', label='cpf')
    plt.xlabel('t (s)')
    plt.ylabel('P (m)')
    # plt.ylim(0.24, 0.34)
    plt.legend()
    save_figs(fig, FIG_DIR_PATH+'pos_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    fig = plt.figure('Velocity est vs base LOCAL frame'+str(idx_exp))
    plt.title('Velocity est\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, b_v_ob_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, b_v_ob_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, b_v_ωb_cpf_arr[:,i], 'rgb'[i]+'--', label='cpf')
    plt.xlabel('t (s)')
    plt.ylabel('V (m/s)')
    plt.legend()
    save_figs(fig, FIG_DIR_PATH+'vel_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    # fig = plt.figure('Velocity est vs mocap BASE frame'+str(idx_exp))
    # plt.title('Velocity est vs mocap\n{}'.format(config))
    # for i in range(3):
    #     plt.plot(t_arr, m_v_wm_arr[:,i], 'rgb'[i], label='est')
    #     plt.plot(t_arr, m_v_wm_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
    #     plt.plot(t_arr, m_v_wm_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    # plt.xlabel('t (s)')
    # plt.ylabel('V (m/s)')
    # plt.legend()
    # save_figs(FIG_DIR_PATH+'vel_base_{}'.format(idx_exp), FILE_TYPES)
    # if CLOSE: plt.close(fig=fig)

    # fig = plt.figure('Velocity lever arm'+str(idx_exp))
    # plt.title('Velocity lever arm\n{}'.format(config))
    # for i in range(3):
    #     plt.plot(t_arr, i_v_lever_arr[:,i], 'rgb'[i], label='est')
    # plt.xlabel('t (s)')
    # plt.ylabel('V (m/s)')
    # plt.legend()
    # save_figs(FIG_DIR_PATH+'vel_lever_{}'.format(idx_exp), FILE_TYPES)
    # if CLOSE: plt.close(fig=fig)


    fig = plt.figure('Orientation est vs mocap'+str(idx_exp))
    plt.title('Orientation est vs mocap\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, w_rpy_m_arr[:,i], 'rgb'[i], label='est')
        plt.plot(t_arr, w_rpy_m_fbk_arr[:,i], 'rgb'[i]+'.', label='fbk')
        plt.plot(t_arr, w_rpy_m_gtr_arr[:,i], 'rgb'[i]+'--', label='moc')
    plt.xlabel('t (s)')
    plt.ylabel('Q')
    plt.legend()
    save_figs(fig, FIG_DIR_PATH+'orientation_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)


    #############
    # ERROR plots
    #############
    title = 'Position error and covariances'
    fig = plt.figure(title+str(idx_exp))
    plt.suptitle(title+'\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k, label='')
        plt.plot(t_arr, w_p_wm_arr[:,k] - w_p_wm_gtr_arr[:,k], 'm', label='est')
        plt.plot(t_arr, w_p_wm_fbk_arr[:,k] - w_p_wm_gtr_arr[:,k], 'c.', label='fbk')
        if COV: 
            plt.plot(tkf_arr,  envel_p[:,k], 'k', label='cov')
            plt.plot(tkf_arr, -envel_p[:,k], 'k', label='cov')
        plt.legend()
    save_figs(fig, FIG_DIR_PATH+'err_pos_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)
    
    # title = 'Velocity error and covariances'
    # fig = plt.figure(title+str(idx_exp))
    # plt.suptitle(title+'\n{}'.format(config))
    # for k in range(3):
    #     plt.subplot(3,1,1+k)
    #     plt.plot(t_arr, w_v_wm_arr[:,k] - w_v_wm_gtr_arr[:,k], 'm', label='est')
    #     plt.plot(t_arr, w_v_wm_fbk_arr[:,k] - w_v_wm_gtr_arr[:,k], 'c.', label='fbk')
    #     if COV: 
    #         plt.plot(tkf_arr,  envel_v[:,k], 'k', label='cov')
    #         plt.plot(tkf_arr, -envel_v[:,k], 'k', label='cov')
    #     plt.legend()
    # save_figs(fig, FIG_DIR_PATH+'err_vel_{}'.format(idx_exp), FILE_TYPES)
    # if CLOSE: plt.close(fig=fig)

    err_o =     np.array([pin.log(w_T_m.rotation*w_T_m_gtr.rotation.T) for w_T_m, w_T_m_gtr in zip(w_T_m_lst, w_T_m_gtr_lst)])
    err_o_fbk = np.array([pin.log(w_T_m.rotation*w_T_m_gtr.rotation.T) for w_T_m, w_T_m_gtr in zip(w_T_m_fbk_lst, w_T_m_gtr_lst)])

    title = 'Orientation error and covariances'
    fig = plt.figure(title+str(idx_exp))
    plt.suptitle(title+'\n{}'.format(config))
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.plot(t_arr, err_o[:,k], 'm', label='est')
        plt.plot(t_arr, err_o_fbk[:,k], 'c.', label='fbk')
        if COV: 
            plt.plot(tkf_arr,  envel_o[:,k], 'k', label='cov')
            plt.plot(tkf_arr, -envel_o[:,k], 'k', label='cov')
        plt.legend()
    save_figs(fig, FIG_DIR_PATH+'err_rot_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    ###############
    # FACTOR ERRORS
    ###############
    title = 'Factor IMU err'
    fig = plt.figure(title+str(idx_exp))
    plt.suptitle(title+'\n{}'.format(config))    
    for k in range(3):
        plt.subplot(3,1,1+k)
        plt.title('POV'[k])
        plt.plot(tkf_arr, fac_imu_err[:,0+3*k], 'r')
        plt.plot(tkf_arr, fac_imu_err[:,1+3*k], 'g')
        plt.plot(tkf_arr, fac_imu_err[:,2+3*k], 'b')
    save_figs(fig, FIG_DIR_PATH+'fac_imu_errors_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    title = 'Factor Pose err'
    fig = plt.figure(title+str(idx_exp))
    plt.suptitle(title+'\n{}'.format(config))    
    for k in range(2):
        plt.subplot(2,1,1+k)
        plt.title('PO'[k])
        plt.plot(tkf_arr, fac_pose_err[:,0+2*k], 'r')
        plt.plot(tkf_arr, fac_pose_err[:,1+2*k], 'g')
        plt.plot(tkf_arr, fac_pose_err[:,2+2*k], 'b')
    save_figs(fig, FIG_DIR_PATH+'fac_pose_errors_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)
    
    ############
    # PARAMETERS
    ############
    title = 'Extrinsics MOCAP'
    fig = plt.figure(title+str(idx_exp))
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
    save_figs(fig, FIG_DIR_PATH+'extr_mocap_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)
    
    title = 'IMU biases'
    fig = plt.figure(title+str(idx_exp))
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
    save_figs(fig, FIG_DIR_PATH+'imu_bias_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    

    #######
    # EXTRA
    #######
    # omg_norm = np.linalg.norm(res_dic['i_omg_oi'], axis=1)
    # title = 'P jumps = f(omg)'
    # fig = plt.figure(title+str(idx_exp))
    # plt.title(title+'\n{}'.format(config))
    # for i in range(3):
    #     plt.plot(omg_norm, o_p_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    # save_figs(FIG_DIR_PATH+'jump_P_fbk_f(gyro)_{}'.format(idx_exp), FILE_TYPES)
    # if CLOSE: plt.close(fig=fig)

    title = 'P jumps = f(t)'
    fig = plt.figure(title+str(idx_exp))
    plt.title(title+'\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, o_p_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    save_figs(fig, FIG_DIR_PATH+'jump_P_fbk_f(t)_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)

    title = 'V jumps = f(t)'
    fig = plt.figure(title+str(idx_exp))
    plt.title(title+'\n{}'.format(config))
    for i in range(3):
        plt.plot(t_arr, o_v_ob_fbk_diff[:,i], 'rgb'[i]+'.')
    save_figs(fig, FIG_DIR_PATH+'jump_V_fbk_f(t)_{}'.format(idx_exp), FILE_TYPES)
    if CLOSE: plt.close(fig=fig)


restore_initial_file(PARAM_FILE)
restore_initial_file(PROC_IMU_PARAM_FILE)
restore_initial_file(TREE_PARAM_FILE)


if SHOW: plt.show()
