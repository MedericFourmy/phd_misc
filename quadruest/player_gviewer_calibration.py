import json
import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import load

from data_readers import read_data_file_laas

TIME_SCALE = 1
NO_CORR = True

calib_file = 'calib.json'
# calib_file = 'calib_kp2_offset.json'
# calib_file = 'calib_kp2_offset_flexi.json'
# calib_file = 'calib_kp2_pert_offset.json'
# calib_file = 'calib_kp2_pert_offset_flexi.json'

with open(calib_file) as json_file:
    calib = json.load(json_file)
for k in calib:
    calib[k] = np.array(calib[k])


m_p_mb = calib['m_p_mb'] if 'm_p_mb' in calib else np.zeros(3)
# m_p_mb = np.array([-0.11,0,-0.015])   # from CAD (imu to base)
m_q_b = calib['m_q_b'] if 'm_q_b' in calib else np.array([0,0,0,1])
m_M_b = pin.XYZQUATToSE3(np.concatenate([m_p_mb, m_q_b]))

w_p_wω = calib['w_p_wω'] if 'w_p_wω' in calib else np.zeros(3)
w_q_ω = calib['w_q_ω'] if 'w_q_ω' in calib else np.array([0,0,0,1]) 
w_M_ω = pin.XYZQUATToSE3(np.concatenate([w_p_wω, w_q_ω]))

# default values if not part of the calibration
offset = calib['offset'] if 'offset' in calib else np.zeros(12)
friction = calib['friction'] if 'friction' in calib else np.zeros(12)
alpha = calib['alpha'] if 'alpha' in calib else np.zeros(12) 

if NO_CORR:
    offset = np.zeros(12)
    friction = np.zeros(12)
    alpha = np.zeros(12)

print('offset', offset)
print('friction', friction)
print('alpha', alpha)


dt = 1e-3

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_in_air_full_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_pointed.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_pointed.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_move_Kp3.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean_slow.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner.npz'


# IRI SECOND TIME
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2_vib02_format.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_test_hysteresis.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_test_hysteresis_vib02.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_test_hysteresis_KD0.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_test_hysteresis_vib02_KD0.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_offset_calibration_format.npz'

file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_format.npz'



robot = load('solo12')
robot.initViewer(loadModel=True, sceneName='world/real')


# gv = robot.viewer.gui
gv = robot.viewer.gui
window_id = 'python-pinocchio'

# arr_dic = read_data_file_laas(file_path, dt)
arr_dic = np.load(file_path)


qa_arr = arr_dic['qa']
dqa_arr = arr_dic['dqa']
tau_arr = arr_dic['tau']
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']


# w_M_m * m_M_b  with m_q_b = identity
w_M_m_lst = [pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m])) * m_M_b 
                       for w_p_wm, w_q_m in zip(w_p_wm_arr, w_q_m_arr)]



# visualize contact points
# Plank dimensions (>= 13/10/21)
width = 0.294
length = 0.389
# height = 0.017

ω_p_ωl_arr = np.array([
    [ length/2,         width/2, 0],
    [ length/2+0.05,  -(width/2+0.05), 0],
    [-length/2,       -(width/2+0.05), 0],
    [ -length/2,       +width/2, 0],
])

w_p_wl_arr = [w_M_ω*ω_p_ωl for ω_p_ωl in ω_p_ωl_arr]

if not NO_CORR:
    for k, pl in enumerate(w_p_wl_arr):
        name = 'hpp-gui/feet{}'.format(k)
        gv.addSphere(name,.01,[1,0,0,1]) 
        gv.applyConfiguration(name,[ pl[0], pl[1], pl[2] ,0,0,0,1])
    gv.refresh()



N = qa_arr.shape[0]
print('N', N)


display_every = 5
display_dt = display_every * dt

for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        qa = qa_arr[i,:]
        dqa = dqa_arr[i,:]
        w_M_wm = w_M_m_lst[i]
        q = np.concatenate([pin.SE3ToXYZQUAT(w_M_wm), qa])

        tau_eff = tau_arr[i,:] - np.sign(dqa)*friction
        qa_corr = qa + offset + alpha*tau_eff
        q[7:] = qa_corr
        robot.display(q)

        time_spent = time.time() - time_start
        if(time_spent < display_dt): time.sleep((display_dt-time_spent)*TIME_SCALE)


