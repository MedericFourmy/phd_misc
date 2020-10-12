import matplotlib.pyplot as plt
from data_readers import read_data_files_mpi, read_data_file_laas, shortened_arr_dic

dt = 1e-3

BASE_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/'

# # folder = "data/solo12_standing_still_2020-10-01_14-22-52/2020-10-01_14-22-52/"
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
data_file = 'Logs_09_10_20_soir/data_2020_10_09_18_58.npz'

arr_dic = read_data_file_laas(BASE_FOLDER+data_file, dt)
# arr_dic = shortened_arr_dic(arr_dic, 2000)

t_arr = arr_dic['t']

S = 0
# S = 2000
t_arr = t_arr[S:]

# 'imu_acc': imu_acc_file, 
# 'o_a_oi': o_a_oi_file, 
# 'i_omg_oi': i_omg_oi_file, 
# 'o_rpy_i': o_rpy_i_file, 
# 'w_pose_wm': w_pose_wm_file, 
# 'm_v_wm': m_v_wm_file, 
# 'w_v_wm': w_v_wm_file, 
# 'qa': qa_file, 
# 'dqa': dqa_file, 
# 'tau': tau_file, 

print('o_R_i')
print(arr_dic['o_R_i'][0,:,:])

if 'imu_acc' in arr_dic:
    plt.figure('imu_acc')
    plt.subplot(3,1,1)
    plt.plot(t_arr, arr_dic['imu_acc'][S:,0], '.', markersize=1)
    plt.subplot(3,1,2)
    plt.plot(t_arr, arr_dic['imu_acc'][S:,1], '.', markersize=1)
    plt.subplot(3,1,3)
    plt.plot(t_arr, arr_dic['imu_acc'][S:,2], '.', markersize=1)

plt.figure('o_a_oi')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['o_a_oi'][S:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['o_a_oi'][S:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['o_a_oi'][S:,2], '.', markersize=1)


plt.figure('i_omg_oi')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['i_omg_oi'][S:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['i_omg_oi'][S:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['i_omg_oi'][S:,2], '.', markersize=1)

plt.figure('o_rpy_i')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['o_rpy_i'][S:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['o_rpy_i'][S:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['o_rpy_i'][S:,2], '.', markersize=1)

plt.figure('w_rpy_m')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['w_rpy_m'][S:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['w_rpy_m'][S:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['w_rpy_m'][S:,2], '.', markersize=1)


plt.figure('qa')
for i in range(12):
    plt.subplot(12,1,i+1)
    plt.plot(t_arr, arr_dic['qa'][S:,i], '.', markersize=1)

plt.figure('dqa')
for i in range(12):
    plt.subplot(12,1,i+1)
    plt.plot(t_arr, arr_dic['dqa'][S:,i], '.', markersize=1)

# plt.figure('o_rpy_i')
# plt.subplot(3,1,1)
# plt.plot(t_arr, arr_dic['o_rpy_i'][S:,0], '.', markersize=1)
# plt.subplot(3,1,2)
# plt.plot(t_arr, arr_dic['o_rpy_i'][S:,1], '.', markersize=1)
# plt.subplot(3,1,3)
# plt.plot(t_arr, arr_dic['o_rpy_i'][S:,2], '.', markersize=1)


plt.show()
