import os
import sys
import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import load

from data_readers import read_data_file_laas

dt = 1e-3

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_in_air_full_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_pointed.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_pointed.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_move_Kp3.npz'  # DIRTY


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean_slow.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner.npz'

# IRI SECOND TIME
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_15_24_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_24_34_format.npz'
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_34_44_format.npz'


m_t_b_mean [-0.1139954  -0.00421959 -0.01335366]
w_quat_ω_mean [-2.90133987e-02  5.65351078e-04  2.38215836e-02  9.99294969e-01]
w_t_ω_mean [-0.02359858 -0.0200456   0.83605627]
w_quat_ω_mean [-2.90133987e-02  5.65351078e-04  2.38215836e-02  9.99294969e-01]
alpha_mean [-0.03923542 -0.14736626  0.21704091 -0.19043326  0.05642432 -0.06556757
 -0.03967905 -0.09597261  0.12296651 -0.0156282  -0.07477355 -0.08113714]


robot = load('solo12')
robot.initViewer(loadModel=True, sceneName='world/plan')

gv = robot.viewer.gui
window_id = 'python-pinocchio'


# arr_dic = read_data_file_laas(file_path, dt)
arr_dic = np.load(file_path)
qa_arr = arr_dic['qa']
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']

N = qa_arr.shape[0]


m_p_mb = np.array([-0.11,0,-0.015])
# w_M_m * m_M_b  with m_q_b = identity
w_p_wm_arr = np.array([pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m])) * m_p_mb 
                       for w_p_wm, w_q_m in zip(w_p_wm_arr, w_q_m_arr)]) 


display_every = 5
display_dt = display_every * dt

for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        qa = qa_arr[i,:]
        w_p_wm = w_p_wm_arr[i,:]
        w_q_m = w_q_m_arr[i,:]
        q = np.concatenate([w_p_wm, w_q_m, qa])
        robot.display(q)

        time_spent = time.time() - time_start
        if(time_spent < display_dt): time.sleep(display_dt-time_spent)


