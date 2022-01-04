import sys
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d


# OUT OF ESTIMATION WOLF
DIRECTORY = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/centroidalkin/figs/window_experiments/'

traj = 'solo_trot_round_feet_format_feet_radius/out_viz1.npz'

arr_dic = np.load(DIRECTORY+traj)

w_pose_m_est_arr = arr_dic['w_pose_m_cpf'][20000:]
w_pose_m_gtr_arr = arr_dic['w_pose_m_gtr'][20000:]

dt = 1e-3
N = w_pose_m_gtr_arr.shape[0]
t_arr = np.arange(N)*dt

# Recover SE3 trajs
w_T_m_est_lst = [pin.XYZQUATToSE3(pose) for pose in w_pose_m_est_arr]
w_T_m_gtr_lst = [pin.XYZQUATToSE3(pose) for pose in w_pose_m_gtr_arr]

Rz = pin.rpy.rpyToMatrix(0, 0, 180)
SIG = 5
aa_gtr = np.array([pin.log(Rz@T.rotation) for T in w_T_m_gtr_lst])
aa_gtr_filt = gaussian_filter1d(aa_gtr, SIG, axis=0)


Naxes = 3
for i in range(Naxes):
    plt.subplot(Naxes,1,i+1)
    plt.plot(t_arr, np.rad2deg(aa_gtr[:,i]), 'b')
    plt.plot(t_arr, np.rad2deg(aa_gtr_filt[:,i]), 'g')
    plt.ylabel('O{} (deg)'.format('xyz'[i]))
plt.xlabel('t (s)')
plt.show()