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


# data_file = 'IRI_10_21/data_2021_10_07_11_40_0.npz'

data_file = 'IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters_format.npz'

file_path = BASE_FOLDER+data_file


# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_with_mocap.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_with_mocap_still.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_move_Kp3.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/selected_qa.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean_slow.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/data_2021_10_15_11_02_0.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_10s.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_1min.npz'

file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground.npz'


arr_dic = read_data_file_laas(file_path, dt)
# arr_dic = np.load(file_path, dt)
# arr_dic = shortened_arr_dic(arr_dic, 23000, 34000)
# arr_dic = shortened_arr_dic(arr_dic, 36000, 45800)

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


def plot_raw(t_arr, arr_dic: np.array, key: str):
    n = arr_dic[key].shape[1]
    c = lambda i: 'rgbk'[i] if n <=4 else ''
    plt.figure(key)
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.plot(t_arr, arr_dic[key][:,i], c(i)+'.', markersize=1)


to_plot = [
    'imu_acc',
    'o_a_oi',
    'i_a_oi',
    'i_omg_oi',
    'o_rpy_i',
    'w_rpy_m',
    'w_q_m',
    'w_p_wm',
    'qa',
    'dqa',
    'tau',
]

for key in to_plot:
    plot_raw(t_arr, arr_dic, key)


plt.show()
