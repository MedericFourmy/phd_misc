import os
import sys
import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import load

from data_readers import read_data_file_laas

MY_NAMING = False
dt = 1e-3

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_in_air_full_format.npz'
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters.npz'


robot = load('solo12')
robot.initViewer(loadModel=True, sceneName='world/plan')

gv = robot.viewer.gui
window_id = 'python-pinocchio'

# arr_dic = read_data_file_laas(file_path, dt)
arr_dic = np.load(file_path)
if MY_NAMING:
    qa_arr = arr_dic['qa']
    w_p_wm_arr = arr_dic['w_p_wm']
    w_q_m_arr = arr_dic['w_q_m']
else:
    qa_arr = arr_dic['q_mes']
    w_p_wm_arr = arr_dic['mocapPosition']
    w_q_m_arr = arr_dic['mocapOrientationQuat']

N = qa_arr.shape[0]


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


