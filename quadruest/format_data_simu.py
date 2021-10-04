import sys
import numpy as np
from data_readers import read_data_traj_dat
import matplotlib.pyplot as plt

traj_path = "/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_gentle_backright_fflyer"
dt = 1e-3
OUT_FILE_NAME = traj_path+'_format.npz'

arr_dic = read_data_traj_dat(traj_path, dt)


# add noise or stuff
arr_dic['imu_acc'] += np.array([0.01, 0.02, 0.03])
arr_dic['i_omg_oi'] += np.array([0.03, 0.02, 0.01])

t_arr = arr_dic['t']


plt.figure('IMU GYR')
plt.title('Raw IMU gyro measurements')
plt.plot(t_arr, arr_dic['i_omg_oi'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,2], 'b', markersize=1)
plt.xlabel('t (s)')
plt.ylabel('omg (rad/s)')


plt.figure('IMU ACC')
plt.title('Raw IMU acc measurements')
plt.plot(t_arr, arr_dic['imu_acc'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,2], 'b', markersize=1)
plt.xlabel('t (s)')
plt.ylabel('a (m/s^2)')

plt.figure('Mocap translation')
plt.plot(t_arr, arr_dic['w_p_wm'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['w_p_wm'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['w_p_wm'][:,2], 'b', markersize=1)

plt.figure('o_a_oi')
plt.plot(t_arr, arr_dic['o_a_oi'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,2], 'b', markersize=1)


np.savez(OUT_FILE_NAME, **arr_dic)
print(OUT_FILE_NAME, 'saved')


if '--show' in sys.argv:
    plt.show()