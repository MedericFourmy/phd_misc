import numpy as np
import pinocchio as pin
from data_readers import read_data_file_laas
import matplotlib.pyplot as plt

# file_name = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_03.npz'
file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_09.npz'  # Standing still (5s), mocap 200Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_10.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_16.npz'  # Moving up (5s), mocap 200Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_17.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_25.npz'  # Standing still (5s), mocap 500Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_26.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_29.npz'  # Moving up (5s), mocap 500Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_30.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_31.npz'  # Moving up->front->down (10s), mocap 500Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_32.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_54.npz'  # Moving up then random movements (15s), mocap 500Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_56.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_57.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_59.npz'  # Already in air, random movements (15s), mocap 500Hz
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_16_03.npz'  # //

dt = 2e-3
dt_mocap = 2e-3
arr_dic = read_data_file_laas(file_name, dt)  # if default format
t_arr = arr_dic['t']
i_omg_oi_arr = arr_dic['i_omg_oi']  # angvel IMU
w_p_wm_arr = arr_dic['w_p_wm']    # position mocap
w_R_m_arr = arr_dic['w_R_m']     # orientation mocap
w_q_m_arr = arr_dic['w_q_m']     # orientation mocap
o_a_oi_arr = arr_dic['o_a_oi']       # IMU absolute acceleration

N = 100


w_R_m_prev_arr = np.roll(w_R_m_arr, N, axis=0)
w_R_m_post_arr = np.roll(w_R_m_arr, -N, axis=0)
# m_omg_om_arr = np.array([   pin.log(w_R_m1.T@w_R_m2)/(2*N*dt_mocap)    for w_R_m1, w_R_m2 in zip(w_R_m_prev_arr, w_R_m_arr)])
m_omg_om_arr = np.array([   pin.log(w_R_m1.T@w_R_m2)/(2*N*dt_mocap)    for w_R_m1, w_R_m2 in zip(w_R_m_prev_arr, w_R_m_post_arr)])

w_p_wm_prev_arr = np.roll(w_p_wm_arr, N, axis=0)
w_p_wm_post_arr = np.roll(w_p_wm_arr, -N, axis=0)


acc_m_arr = np.array([  (w_p_wm0 + w_p_wm2 - 2*w_p_wm1)/((N*dt_mocap)**2)      
                        for w_p_wm0, w_p_wm1, w_p_wm2 in zip(w_p_wm_prev_arr, w_p_wm_arr, w_p_wm_post_arr)])


plt.figure('OMG')
for i in range(3):
    plt.plot(t_arr, i_omg_oi_arr[:,i], 'rgb'[i], label='IMU')
    plt.plot(t_arr, m_omg_om_arr[:,i], 'rgb'[i]+'.', label='MOCAP')
plt.legend()

plt.figure('ACC')
plt.plot(t_arr, np.linalg.norm(o_a_oi_arr,  axis=1), 'b.', label='IMU abs acc')
plt.plot(t_arr, np.linalg.norm(acc_m_arr, axis=1), 'r.',  label='MOCAP abs acc')
plt.legend()


############################################################
# Compute accelerations in imu referential  
############################################################
i_a_oi_arr = arr_dic['i_a_oi']       # IMU absolute acceleration
m_p_mi = [0.1163, 0.0, 0.02]
m_q_i = [0.0, 0.0, 0.0, 1.0]

w_p_wi_arr = np.array([w_p_wm + w_R_m@m_p_mi  for w_p_wm, w_R_m in zip(w_p_wm_arr, w_R_m_arr)])
w_p_wi_prev_arr = np.roll(w_p_wi_arr, N, axis=0)
w_p_wi_post_arr = np.roll(w_p_wi_arr, -N, axis=0)

w_acc_i_arr = np.array([  (w_p_wi0 + w_p_wi2 - 2*w_p_wi1)/((N*dt_mocap)**2)      
                        for w_p_wi0, w_p_wi1, w_p_wi2 in zip(w_p_wi_prev_arr, w_p_wi_arr, w_p_wi_post_arr)])

i_acc_i_arr = np.array([w_R_m.T @ w_acc_i for w_acc_i, w_R_m in zip(w_acc_i_arr, w_R_m_arr)])

plt.figure('Mocap position')
for i in range(3):
    plt.plot(t_arr, w_p_wm_arr[:,i], 'rgb'[i], label='posi mocap frame')
    plt.plot(t_arr, w_p_wi_arr[:,i], 'rgb'[i]+'--', label='posi imu frame')
plt.legend()


plt.figure('ACC IMU frame')
for i in range(3):
    plt.plot(t_arr, i_a_oi_arr[:,i], 'rgb'[i], label='IMU abs acc')
    plt.plot(t_arr, i_acc_i_arr[:,i],  'rgb'[i]+'.',  label='MOCAP abs acc')
plt.legend()

imu_acc_arr = arr_dic['imu_acc']
plt.figure('IMU acc meas')
for i in range(3):
    plt.plot(t_arr, imu_acc_arr[:,i], 'rgb'[i], label='IMU acc meas')
# plt.legend()



plt.show()