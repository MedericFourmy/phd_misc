import os
import sys
import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import load

from data_readers import read_data_file_laas

dt = 1e-3
SLEEP = False

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
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min_34_44_format.npz'


# OUT OF ESTIMATION WOLF
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out0.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out1.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out2.npz'
color = [1,0,0,1]


# robot = load('solo12')

URDF_NAME = 'solo12_pointed_feet.urdf'
path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/' + URDF_NAME
srdf = path + '/srdf/solo.srdf'
robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())




robot.initViewer(loadModel=True, sceneName='world/plan')

gv = robot.viewer.gui
window_id = 'python-pinocchio'


# arr_dic = read_data_file_laas(file_path, dt)
arr_dic = np.load(file_path)
qa_arr = arr_dic['qa']
w_p_wm_arr = arr_dic['o_p_ob']
w_q_m_arr = arr_dic['o_q_b']
# w_p_wm_arr = arr_dic['o_p_ob_fbk']
# w_q_m_arr = arr_dic['o_q_b_fbk']

N = qa_arr.shape[0]


LEGS = ['FL', 'FR', 'HL', 'HR']
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

display_every = 500
display_dt = display_every * dt

out_name = file_path.split('/')[-1].split('.')[0]

for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        qa = qa_arr[i,:]
        w_p_wm = w_p_wm_arr[i,:]
        w_q_m = w_q_m_arr[i,:]
        q = np.concatenate([w_p_wm, w_q_m, qa])
        robot.display(q)

        for k, cid in enumerate(cids):
            pl = robot.framePlacement(q, cid, update_kinematics=True).translation

            name = 'hpp-gui/feet_{}_{}_{}'.format(out_name, LEGS[k], i)
            gv.addSphere(name, .001, color) 
            gv.applyConfiguration(name,[ pl[0], pl[1], pl[2] ,0,0,0,1])
        gv.refresh()


        time_spent = time.time() - time_start
        if(SLEEP and time_spent < display_dt): time.sleep(display_dt-time_spent)


