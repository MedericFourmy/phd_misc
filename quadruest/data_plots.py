import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from data_readers import read_data_files_mpi, read_data_file_laas, shortened_arr_dic

dt = 1e-3

BASE_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/'

# folder = "data/solo12_standing_still_2020-10-01_14-22-52/2020-10-01_14-22-52/"
# # folder = "data/solo12_com_oscillation_2020-10-01_14-22-13/2020-10-01_14-22-13/"
# folder = "data/solo12_stamping_2020-09-29_18-04-37/2020-09-29_18-04-37/"
# # arr_dic = read_data_files_mpi(folder, dt)  # if default format
# arr_dic = read_data_files_mpi(folder, dt, delimiter=',')  # with "," delimiters

# data_file = 'data.npz'
# data_file = 'data_2020_10_08_09_50_Walking_Novicon.npz'
# data_file = 'data_2020_10_08_10_04_StandingStill.npz'
# data_file = 'data_2020_10_09_16_10_Stamping.npz'
# data_file = 'data_2020_10_09_16_12_SinTraj.npz'

# last data with turned imu
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_18_58.npz'

# data_file = "Logs_15_10_2020/data_2020_10_15_14_34.npz"  # standing 
# data_file = "Logs_15_10_2020/data_2020_10_15_14_36.npz"    # sinXYZ

# data_file = "data_2020_10_31_20_12.npz"
# data_file = "Logs_15_10_2020/data_2020_10_15_14_38.npz"


data_file = 'IRI_10_21/data_2021_10_07_11_40_0.npz'



arr_dic = read_data_file_laas(BASE_FOLDER+data_file, dt)
# arr_dic = shortened_arr_dic(arr_dic, 2000, len(arr_dic['t'])-200)
# arr_dic = shortened_arr_dic(arr_dic, 5000, 7150)

t_arr = arr_dic['t']

i_a_oi_arr = arr_dic['i_a_oi']
imu_acc_arr = arr_dic['imu_acc']
o_R_i_arr = arr_dic['o_R_i']

g = 9.806*np.array([0,0,-1])
# try to compute imu biases
bacc_arr = np.array([imu_acc + o_R_i.T@g - i_a_oi for i_a_oi, imu_acc, o_R_i in zip(i_a_oi_arr, imu_acc_arr, o_R_i_arr)])

plt.figure('Bias ACC')
plt.plot(t_arr, bacc_arr[:,0], 'r.', markersize=1)
plt.plot(t_arr, bacc_arr[:,1], 'g.', markersize=1)
plt.plot(t_arr, bacc_arr[:,2], 'b.', markersize=1)

plt.figure('imu_acc')
plt.subplot(3,1,1)
plt.plot(t_arr, imu_acc_arr[:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, imu_acc_arr[:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, imu_acc_arr[:,2], '.', markersize=1)

plt.figure('o_a_oi')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['o_a_oi'][:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['o_a_oi'][:,2], '.', markersize=1)

plt.figure('i_a_oi')
plt.subplot(3,1,1)
plt.plot(t_arr, i_a_oi_arr[:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, i_a_oi_arr[:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, i_a_oi_arr[:,2], '.', markersize=1)

plt.figure('i_omg_oi')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,2], '.', markersize=1)

plt.figure('o_rpy_i')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['o_rpy_i'][:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['o_rpy_i'][:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['o_rpy_i'][:,2], '.', markersize=1)

plt.figure('w_rpy_m')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['w_rpy_m'][:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['w_rpy_m'][:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['w_rpy_m'][:,2], '.', markersize=1)


plt.figure('qa')
for i in range(12):
    plt.subplot(12,1,i+1)
    plt.plot(t_arr, arr_dic['qa'][:,i], '.', markersize=1)

plt.figure('dqa')
for i in range(12):
    plt.subplot(12,1,i+1)
    plt.plot(t_arr, arr_dic['dqa'][:,i], '.', markersize=1)

# plt.figure('o_rpy_i')
# plt.subplot(3,1,1)
# plt.plot(t_arr, arr_dic['o_rpy_i'][:,0], '.', markersize=1)
# plt.subplot(3,1,2)
# plt.plot(t_arr, arr_dic['o_rpy_i'][:,1], '.', markersize=1)
# plt.subplot(3,1,3)
# plt.plot(t_arr, arr_dic['o_rpy_i'][:,2], '.', markersize=1)


plt.show()
