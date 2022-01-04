import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../quadruest') )
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas, shortened_arr_dic


file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/solo_resting_zeros.npz'

arr_dic = read_data_file_laas(file_path, 1e-3)
# shortened_arr_dic(arr_dic, S=15000, N=20000)

robot = load('solo12')

t_arr = arr_dic['t']
qa_arr = arr_dic['qa']
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']
N = t_arr.shape[0]
 
m_p_mb = np.array([0,0,-0.015])

# w_M_m * m_M_b  with m_q_b = identity
w_p_wm_arr = np.array([pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m])) * m_p_mb 
                       for w_p_wm, w_q_m in zip(w_p_wm_arr, w_q_m_arr)]) 


LEGS = ['FL', 'FR', 'HL', 'HR']
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

feet_posi_arr = np.zeros((N, 12))
q = np.zeros(19)
for i, (qa, w_p_wm, w_q_m)  in enumerate(zip(qa_arr, w_p_wm_arr, w_q_m_arr)):
    q[:3] = w_p_wm
    q[3:7] = w_q_m
    q[7:] = qa
    for j, cid in enumerate(cids):
        pi = robot.framePlacement(q, cid, update_kinematics=True).translation
        feet_posi_arr[i,3*j:3*j+3] = pi


for j in range(4):
    plt.figure(LEGS[j])
    for k in range(3):
        plt.plot(t_arr, feet_posi_arr[:,3*j+k], 'rgb'[k])
    plt.hlines(0, t_arr[0], t_arr[-1], 'b')

plt.show()