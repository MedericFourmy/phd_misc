import numpy as np
import pinocchio as pin
from data_readers import read_data_file_laas
import matplotlib.pyplot as plt

# file_name = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_2021_03_16/data_2021_03_16_18_03.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_09.npz'  # Standing still (5s), mocap 200Hz
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

# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_24.npz'  # nomove 5 minutes
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54.npz'  # //
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56.npz'  # //

# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_2_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_3_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54_4_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_3_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_4_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56_5_format.npz'  # small jump





# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_16_48_calib_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_04_calib_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_08_calib_format.npz'  ######
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_11_calib_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_16_calib_format.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_19_calib_format.npz'

# POINTFEET DATA
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_36.npz'
# file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_38.npz'
file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_39.npz'



# dt = 2e-3
# dt = 2e-3


dt = 1e-3   # sample data
arr_dic = read_data_file_laas(file_name, dt)  # if default format
# arr_dic = np.load(file_name)
t_arr = arr_dic['t']
i_omg_oi_arr = arr_dic['i_omg_oi']  # angvel IMU
w_p_wm_arr = arr_dic['w_p_wm']    # position mocap
w_R_m_arr = arr_dic['w_R_m']     # orientation mocap
w_q_m_arr = arr_dic['w_q_m']     # orientation mocap
o_a_oi_arr = arr_dic['o_a_oi']       # IMU absolute acceleration

# forgot to interpolate the rotation matrix as well
w_R_m_arr = np.array([pin.Quaternion(q.reshape((4,1))).toRotationMatrix() for q in w_q_m_arr])

# to smoothen differentiation and get rid of the fact that 
# mocap data is over-sampled from 200Hz (~) 
N = 100  


w_R_m_prev_arr = np.roll(w_R_m_arr, N, axis=0)
w_R_m_post_arr = np.roll(w_R_m_arr, -N, axis=0)
# m_omg_om_arr = np.array([   pin.log(w_R_m1.T@w_R_m2)/(2*N*dt)    for w_R_m1, w_R_m2 in zip(w_R_m_prev_arr, w_R_m_arr)])
m_omg_om_arr = np.array([   pin.log(w_R_m1.T@w_R_m2)/(2*N*dt)    for w_R_m1, w_R_m2 in zip(w_R_m_prev_arr, w_R_m_post_arr)])

w_p_wm_prev_arr = np.roll(w_p_wm_arr, N, axis=0)
w_p_wm_post_arr = np.roll(w_p_wm_arr, -N, axis=0)


acc_m_arr = np.array([  (w_p_wm0 + w_p_wm2 - 2*w_p_wm1)/((N*dt)**2)      
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

w_acc_i_arr = np.array([  (w_p_wi0 + w_p_wi2 - 2*w_p_wi1)/((N*dt)**2)      
                        for w_p_wi0, w_p_wi1, w_p_wi2 in zip(w_p_wi_prev_arr, w_p_wi_arr, w_p_wi_post_arr)])

i_acc_i_arr = np.array([w_R_m.T @ w_acc_i for w_acc_i, w_R_m in zip(w_acc_i_arr, w_R_m_arr)])

plt.figure('Mocap position')
for i in range(3):
    plt.plot(t_arr, w_p_wm_arr[:,i], 'rgb'[i], label='posi mocap frame')
    plt.plot(t_arr, w_p_wi_arr[:,i], 'rgb'[i]+'--', label='posi imu frame')
plt.legend()

plt.figure('Mocap quaternion')
for i in range(4):
    plt.plot(t_arr, w_q_m_arr[:,i], 'rgbk'[i], label='quat mocap frame')
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