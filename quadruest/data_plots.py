import numpy as np
import pinocchio as pin
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
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_18_58.npz'

# data_file = "Logs_15_10_2020/data_2020_10_15_14_34.npz"  # standing 
data_file = "Logs_15_10_2020/data_2020_10_15_14_36.npz"    # sinXYZ

# data_file = "data_2020_10_31_20_12.npz"
# data_file = "Logs_15_10_2020/data_2020_10_15_14_38.npz"


arr_dic = read_data_file_laas(BASE_FOLDER+data_file, dt)
# arr_dic = shortened_arr_dic(arr_dic, 2000, len(arr_dic['t'])-200)
# arr_dic = shortened_arr_dic(arr_dic, 5000, 7150)

t_arr = arr_dic['t']


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
#  

i_a_oi_arr = np.array([o_R_i.T@o_a_oi for o_R_i, o_a_oi in zip(arr_dic['o_R_i'], arr_dic['o_a_oi'])])


o_qvec_i_wolf = np.array([0.0035, -0.0122, 0.68683, 0.72673])
o_q_i_wolf = pin.Quaternion(o_qvec_i_wolf.reshape((4,1)))
o_R_i_wolf = o_q_i_wolf.toRotationMatrix()
imu_acc_rot = (o_R_i_wolf@arr_dic['imu_acc'].T).T
print('imu_acc_rot mean/cov')
print(np.mean(imu_acc_rot, axis=0))
print('ACC std')
print(np.sqrt(np.cov(imu_acc_rot, rowvar=False)))
print('i_omg_oi mean/cov')
print(np.mean(arr_dic['i_omg_oi'], axis=0))
print('GYR std')
print(np.sqrt(np.cov(arr_dic['i_omg_oi'], rowvar=False)))

imu_acc_norm = np.linalg.norm(arr_dic['imu_acc'], axis=1)
cov_imu_acc = np.cov(arr_dic['imu_acc'], rowvar=False)
cov_o_a_oi = np.cov(arr_dic['o_a_oi'], rowvar=False)
print('cov_imu_acc')
print(cov_imu_acc)
print('cov_o_a_oi')
print(cov_o_a_oi)

print('imu_acc_norm mean: ', imu_acc_norm.mean())

print('o_R_i')
print(arr_dic['o_R_i'][0,:,:])

plt.figure('imu_acc')
plt.subplot(3,1,1)
plt.plot(t_arr, arr_dic['imu_acc'][:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, arr_dic['imu_acc'][:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, arr_dic['imu_acc'][:,2], '.', markersize=1)

plt.figure('imu_acc norm')
plt.plot(t_arr[2000:-1000], imu_acc_norm[2000:-1000], '.', markersize=1)

plt.figure('imu_acc_rot')
plt.subplot(3,1,1)
plt.plot(t_arr, imu_acc_rot[:,0], '.', markersize=1)
plt.subplot(3,1,2)
plt.plot(t_arr, imu_acc_rot[:,1], '.', markersize=1)
plt.subplot(3,1,3)
plt.plot(t_arr, imu_acc_rot[:,2], '.', markersize=1)

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
