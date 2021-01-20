import sys
import numpy as np
from scipy import signal
import pinocchio as pin
from matplotlib import pyplot as plt
from data_readers import read_data_file_laas, read_data_files_mpi, shortened_arr_dic
from contact_forces_estimator import ContactForcesEstimator
from example_robot_data import loadSolo

# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_34.npz'  # standing still
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_36.npz'    # sinXYZ
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_38.npz'  # stamping

# data_2020_10_15_18_21: ...
# data_2020_10_15_18_23: walk
# data_2020_10_15_18_24: walk
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Log_15_10_2020_part2/data_2020_10_15_18_21.npz'
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Log_15_10_2020_part2/data_2020_10_15_18_23.npz'
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Log_15_10_2020_part2/data_2020_10_15_18_24.npz'

# data_2020_10_29_18_03 walk 0.32 s, no mocap 
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/data_2020_10_29_18_03.npz'
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_38.npz'

# walking and stoping data
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_05_10_2020_18h/data_2020_11_05_18_18.npz'  # standing still+walk 0.48 FAIL
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_05_10_2020_18h/data_2020_11_05_18_20.npz'  # standing still+walk 0.48

# OUT_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_05_10_2020_18h/data_2020_11_05_18_20_format_forces.npz'

IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Walk_17_12_2020/data_2020_12_17_14_25.npz'
# IN_FILE_NAME = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Walk_17_12_2020/data_2020_12_17_14_29.npz'
OUT_FILE_NAME = IN_FILE_NAME.split('.')[0]+'_format.npz'

THRESH_VIZ = 8

SAVE = True


dt = 1e-3  # discretization timespan
arr_dic = read_data_file_laas(IN_FILE_NAME, dt)
N = len(arr_dic['t'])
arr_dic = shortened_arr_dic(arr_dic, 0, N=N-200)
# arr_dic = shortened_arr_dic(arr_dic, 0, 2000)
# arr_dic = shortened_arr_dic(arr_dic, 2000, -200)
t_arr = arr_dic['t']
N = len(t_arr)


# mean the forces
# arr_dic['o_a_oi'][:,:] = np.mean(arr_dic['o_a_oi'], axis=0)
# arr_dic['imu_acc'][:,:] = np.mean(arr_dic['imu_acc'], axis=0)
# arr_dic['i_omg_oi'][:,:] = np.mean(arr_dic['i_omg_oi'], axis=0)
# arr_dic['qa'][:,:] = np.mean(arr_dic['qa'], axis=0)

# filter the kinematics with centered window to compute 1rst and 2nd derivatives
dqa_filt_arr = signal.savgol_filter(arr_dic['qa'], window_length=25, polyorder=2, deriv=1, delta=dt, mode='mirror', axis=0)
ddqa_filt_arr = signal.savgol_filter(arr_dic['qa'], window_length=25, polyorder=2, deriv=2, delta=dt, mode='mirror', axis=0)

# compute angular acceleration
o_R_i_arr = arr_dic['o_R_i']
i_omg_oi_arr = arr_dic['i_omg_oi']
o_omg_i_arr = np.array([o_R_i@i_omg_oi for o_R_i, i_omg_oi in zip(o_R_i_arr, i_omg_oi_arr)])
o_domg_i_arr = signal.savgol_filter(o_omg_i_arr, window_length=25, polyorder=2, deriv=1, delta=dt, mode='mirror', axis=0)

i_domg_i_arr = np.array([o_R_i.T@o_domg_i for o_R_i, o_domg_i in zip(o_R_i_arr, o_domg_i_arr)])

rpy_arr = np.array([pin.rpy.matrixToRpy(o_R_i) for o_R_i in o_R_i_arr])

# Now compute the forces using the robot model
robot = loadSolo(False)
robot.model.gravity.linear = np.array([0, 0, -9.806])
LEGS = ['FL', 'FR', 'HL', 'HR']
nb_feet = len(LEGS)
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

cforces_est = ContactForcesEstimator(robot, contact_frame_ids, dt)

o_forces_arr = np.zeros((N,12))
l_forces_arr = np.zeros((N,12))
o_forces_sum = np.zeros((N,3))
detect_arr = np.zeros((N,4))
for i in range(N):

    qa = arr_dic['qa'][i,:]
    dqa = dqa_filt_arr[i,:]
    ddqa = ddqa_filt_arr[i,:]*0
    o_R_i = o_R_i_arr[i,:,:]
    i_omg_oi = i_omg_oi_arr[i,:]
    i_domg_i = i_domg_i_arr[i,:]
    o_a_oi = arr_dic['o_a_oi'][i,:]
    tauj = arr_dic['tau'][i,:]

    o_forces = cforces_est.compute_contact_forces2(qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_i, o_a_oi, tauj, world_frame=True)
    l_forces = cforces_est.compute_contact_forces2(qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_i, o_a_oi, tauj, world_frame=False)

    o_forces_sum[i,:] = np.sum(o_forces, axis=0)

    o_forces_arr[i,:] = o_forces.flatten()
    l_forces_arr[i,:] = l_forces.flatten()

    detect_arr[i,:] = (o_forces[:,2] > THRESH_VIZ)

# store forces
arr_dic['l_forces'] = l_forces_arr


if SAVE:
    np.savez(OUT_FILE_NAME, **arr_dic)
    print(OUT_FILE_NAME, ' saved')

q = robot.model.referenceConfigurations['standing']
robot.com(q)
m = robot.data.mass[0]
g = robot.model.gravity


o_forces_sum[:,:] = np.mean(o_forces_sum, axis=0)


plt.figure('o f sum')
plt.plot(t_arr, o_forces_sum[:,0], 'r.', markersize=1)
plt.plot(t_arr, o_forces_sum[:,1], 'g.', markersize=1)
plt.plot(t_arr, o_forces_sum[:,2], 'b.', markersize=1)
plt.hlines(0, t_arr[0]-1, t_arr[-1]+1, 'k')
plt.hlines(-m*g.linear[2], t_arr[0]-1, t_arr[-1]+1, 'k')

plt.figure('o_a_oi')
plt.plot(t_arr, arr_dic['o_a_oi'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,2], 'b', markersize=1)


plt.figure('IMU ACC')
plt.plot(t_arr, arr_dic['imu_acc'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,2], 'b', markersize=1)

plt.figure('RPY')
plt.plot(t_arr, rpy_arr[:,0], 'r', markersize=1)
plt.plot(t_arr, rpy_arr[:,1], 'g', markersize=1)
plt.plot(t_arr, rpy_arr[:,2], 'b', markersize=1)



plt.figure('o forces for each leg')
NL = 4
for k in range(NL):
    plt.subplot(NL,1,k+1)
    # plt.plot(t_arr, o_forces_arr[:,3*k+0], 'r', markersize=1)
    # plt.plot(t_arr, o_forces_arr[:,3*k+1], 'g', markersize=1)
    plt.plot(t_arr, o_forces_arr[:,3*k+2], 'b', markersize=1)
    plt.plot(t_arr, detect_arr[:,k]*THRESH_VIZ, 'k')
    plt.plot(t_arr, arr_dic['contactStatus'][:,k]*THRESH_VIZ*0.9, 'r', markersize=1)

# fig, axs = plt.subplots(1,1, figsize=(6,2.2))
# fig.canvas.set_window_title('Forces solo leg 1')
# axs.plot(t_arr, o_forces_arr[:,0], 'r.', markersize=1)
# axs.plot(t_arr, o_forces_arr[:,1], 'g.', markersize=1)
# axs.plot(t_arr, o_forces_arr[:,2], 'b.', markersize=1)
# axs.set_xlabel('time [s]')
# axs.set_ylabel('forces [N]')
# axs.yaxis.set_label_position("right")
# axs.grid(True)
# fig.savefig('forces_solo_1leg.pdf')

if '--show' in sys.argv:
    plt.show()