import numpy as np
import pandas as pd
import pinocchio as pin

import matplotlib.pyplot as plt


# file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020/data_2020_11_30_15_15.npz'
# file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020/data_2020_11_30_15_16.npz'
# file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020/data_2020_11_30_15_17.npz'
# file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_17.npz'  # stamping
file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# file_traj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_22.npz'  # walking

file_qa_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_q.dat'
file_va_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_v.dat'
file_tau_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_tau.dat'
# file_qa_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_stamping_q.dat'
# file_va_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_stamping_v.dat'
# file_tau_traj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_stamping_tau.dat'


df_qa_plan = pd.read_csv(file_qa_traj_plan, sep=' ', header=None)
df_va_plan = pd.read_csv(file_va_traj_plan, sep=' ', header=None)
df_tau_plan = pd.read_csv(file_tau_traj_plan, sep=' ', header=None)
qa_plan_arr = df_qa_plan.to_numpy()[:,1:]  # remove index and switch to array
va_plan_arr = df_va_plan.to_numpy()[:,1:]  # remove index and switch to array
tau_plan_arr = df_tau_plan.to_numpy()[:,1:]  # remove index and switch to array

# file_traj_control_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_control_2020_11_30_17_22.npz'  # walking
# arr_dic_control = np.load(file_traj_control_real)
# qa_plan_arr = arr_dic_control['log_qdes'].T
# va_plan_arr = arr_dic_control['log_vdes'].T
# tau_plan_arr = arr_dic_control['log_tau_ff'].T


arr_dic = np.load(file_traj_real)
qa_real_arr = arr_dic['q_mes']
va_real_arr = arr_dic['v_mes']
tau_real_arr = arr_dic['torquesFromCurrentMeasurment']

Kp = 10
Kd = 0.5
# Kp = 3
# Kd = 0.2
tau_pdp_arr = Kp*(qa_plan_arr - qa_real_arr) + Kd*(va_plan_arr - va_real_arr) + tau_plan_arr

N = tau_plan_arr.shape[0]
Nbis = tau_real_arr.shape[0]
dt = 1e-3
t_arr = np.arange(N-10)*dt

print(N)
print(Nbis)


plt.figure('1-6')
NFEET = 6
for i in range(NFEET):
    plt.subplot(NFEET,1,1+i)
    plt.plot(t_arr, tau_plan_arr[:-10,i], 'r', label='ffwd')
    plt.plot(t_arr, tau_real_arr[:-10,i], 'g', label='meas')
    plt.plot(t_arr, tau_pdp_arr[:-10,i], 'b', label='pdp')
    plt.legend()

plt.figure('7-12')
NFEET = 6
plt
for i in range(NFEET):
    plt.subplot(NFEET,1,1+i)
    plt.plot(t_arr, tau_plan_arr[:-10,i+6], 'r', label='ffwd')
    plt.plot(t_arr, tau_real_arr[:-10,i+6], 'g', label='meas')
    plt.plot(t_arr, tau_pdp_arr[:-10,i+6], 'b', label='pdp')
    plt.legend()

plt.show()