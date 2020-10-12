import numpy as np
import pinocchio as pin
from data_readers import read_data_files_mpi, read_data_file_laas, shortened_arr_dic


DATA_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/'
dt = 1e-3
# 18:58 Fixe 4 stance phase (20s)
# 19:00 Rotation 4 stance phase (30s)
# 19:02 Mouvement avant arrière, rotation, rotation, mvt bas haut, roll (30s)
# 19:03 Replay sin wave
# 19:05 Replay stamping
# 19:06 Marche 0.32 (30s)
data_file_lst = ['Logs_09_10_20_soir/data_2020_10_09_18_58.npz',
                 'Logs_09_10_20_soir/data_2020_10_09_19_00.npz',
                 'Logs_09_10_20_soir/data_2020_10_09_19_02.npz',
                 'Logs_09_10_20_soir/data_2020_10_09_19_03.npz',
                 'Logs_09_10_20_soir/data_2020_10_09_19_05.npz',
                 'Logs_09_10_20_soir/data_2020_10_09_19_06.npz',]
# data_file_lst = [
#     'Logs_09_10_20_soir/data_2020_10_09_19_02.npz',
# ]

for data_file in data_file_lst:
    arr_dic = read_data_file_laas(DATA_FOLDER+data_file, dt)
    N = len(arr_dic['t'])

    err_max_m_R_i = np.zeros(3)
    err_max_w_R_o = np.zeros(3)
    o_rpy_i_arr = np.zeros((N,3))
    w_rpy_m_arr = np.zeros((N,3))
    for i in range(N):
        o_R_i = pin.Quaternion(arr_dic['o_q_i'][i,:].reshape((4,1))).toRotationMatrix()
        w_R_m = pin.Quaternion(arr_dic['w_q_m'][i,:].reshape((4,1))).toRotationMatrix()
        # o_R_i = arr_dic['o_R_i'][i,:,:]
        # w_R_m = arr_dic['w_R_m'][i,:,:]
        o_rpy_i = pin.rpy.matrixToRpy(o_R_i)
        w_rpy_m = pin.rpy.matrixToRpy(w_R_m)
        o_rpy_i_arr[i,:] = np.rad2deg(o_rpy_i)
        w_rpy_m_arr[i,:] = np.rad2deg(w_rpy_m)
        m_R_i = w_R_m.T@o_R_i
        w_R_o = w_R_m@o_R_i.T
        err_m_R_i = np.rad2deg(pin.log3(m_R_i))
        err_w_R_o = np.rad2deg(pin.log3(w_R_o))
        if np.linalg.norm(err_m_R_i) > np.linalg.norm(err_max_m_R_i):
            err_max_m_R_i = err_m_R_i.copy()
        if np.linalg.norm(err_w_R_o) > np.linalg.norm(err_max_w_R_o):
            err_max_w_R_o = err_w_R_o.copy()
    print()
    print(data_file, 'err_o_deg')
    print('err_max_m_R_i: ', err_max_m_R_i)
    print('err_max_w_R_o: ', err_max_w_R_o)


    plt.figure(data_file+' DEG error')
    for j in range(3):
        plt.subplot(3,1,1+j)
        plt.plot(arr_dic['t'], o_rpy_i_arr[:,j], 'r', label='IMU')
        plt.plot(arr_dic['t'], w_rpy_m_arr[:,j], 'b', label='MOCAP')
        plt.legend()
plt.show()
