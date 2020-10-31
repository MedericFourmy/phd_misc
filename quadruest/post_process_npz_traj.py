import os
import sys
import numpy as np
import datetime
from scipy import signal
import pinocchio as pin
import matplotlib
PGF = False
if PGF:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

# legend = {'legend.fontsize': 20,
#           'legend.handlelength': 2}
# legend = {'legend.fontsize': 13}
# matplotlib.rcParams.update(legend)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# uses https://github.com/uzh-rpg/rpg_trajectory_evaluation.git
sys.path.append('/home/mfourmy/Documents/Phd_LAAS/installations/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation') 
import trajectory as rpg_traj

from data_readers import shortened_arr_dic


DATA_FOLDER_RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/'

# 18:58 Fixe 4 stance phase (20s)
# 19:00 Rotation 4 stance phase (30s)
# 19:02 Mouvement avant arrière, rotation, rotation, mvt bas haut, roll (30s)
# 19:03 Replay sin wave
# 19:05 Replay stamping
# 19:06 Marche 0.32 (30s)
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_18_58.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_00.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_02.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_03.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_05.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_06.npz'

# from wolf estimation
data_file = 'out.npz'

# Keys:
print('Reading ', DATA_FOLDER_RESULTS+data_file)
res_arr_dic = np.load(DATA_FOLDER_RESULTS+data_file)
# res_arr_dic = shortened_arr_dic(res_arr_dic, 1000, -1000)

dt = 1e-3
t_arr = res_arr_dic['t']
N = len(t_arr)
print('N: ', N)
# GT
w_p_wm_arr = res_arr_dic['w_p_wm']
w_q_m_arr = res_arr_dic['w_q_m']
w_v_wm_arr = res_arr_dic['w_v_wm']
m_v_wm_arr = res_arr_dic['m_v_wm']

o_q_b_arr = res_arr_dic['o_q_b']
o_p_ob_arr = res_arr_dic['o_p_ob']
o_v_ob_arr = res_arr_dic['o_v_ob']


POVCDL = 'o_p_oc' in res_arr_dic
o_p_oc_arr = 42*np.ones((N,3))
o_v_oc_arr = 42*np.ones((N,3))
b_v_oc_arr = 42*np.ones((N,3))

if POVCDL:
    o_p_oc_arr = res_arr_dic['o_p_oc']
    o_v_oc_arr = res_arr_dic['o_v_oc']


# use IMU orientation instead of Mo-Cap to compare with estimator
# o_p_ob_arr = np.zeros((N,3))
# w_p_wm_arr = np.zeros((N,3))
# w_q_m_arr = res_arr_dic['o_q_i']


#####################################
# MOCAP freq 200Hz vs robot freq 1kHz
NB_OVER = 5

# compute filterd Mo-Cap velocity
w_p_wm_arr_sub = w_p_wm_arr[::NB_OVER]
w_v_wm_arr_sagol_sub = signal.savgol_filter(w_p_wm_arr_sub, window_length=21, polyorder=2, deriv=1, axis=0, delta=NB_OVER*dt, mode='mirror')
w_v_wm_arr_sagol = w_v_wm_arr_sagol_sub.repeat(NB_OVER, axis=0) 

# plt.figure('Mo-Cap velocity filtering')
# plt.subplot(3,1,1)
# plt.plot(t_arr, w_v_wm_arr[:,0], 'r.', markersize=1, label='vx numdiff')
# # plt.plot(t_arr, w_v_wm_filtered_arr[:,0], 'b.', markersize=1, label='vx filtered')
# plt.plot(t_arr, w_v_wm_arr_sagol[:,0], 'k.', markersize=1, label='vx filtered better')
# plt.subplot(3,1,2)
# plt.plot(t_arr, w_v_wm_arr[:,1], 'r.', markersize=1, label='vy numdiff')
# # plt.plot(t_arr, w_v_wm_filtered_arr[:,1], 'b.', markersize=1, label='vy filtered')
# plt.plot(t_arr, w_v_wm_arr_sagol[:,1], 'k.', markersize=1, label='vx filtered better')
# plt.subplot(3,1,3)
# plt.plot(t_arr, w_v_wm_arr[:,2], 'r.', markersize=1, label='vz numdiff')
# # plt.plot(t_arr, w_v_wm_filtered_arr[:,2], 'b.', markersize=1, label='vz filtered')
# plt.plot(t_arr, w_v_wm_arr_sagol[:,2], 'k.', markersize=1, label='vz filtered better')
# plt.legend()
#####################

# compute base velocities
o_R_b_lst = [pin.Quaternion(o_q_b.reshape((4,1))).toRotationMatrix() for o_q_b in o_q_b_arr] 
w_R_m_lst = [pin.Quaternion(w_q_m.reshape((4,1))).toRotationMatrix() for w_q_m in w_q_m_arr] 
# m_v_wm_arr_filt = np.array([w_R_m.T @ w_v_wm for w_R_m, w_v_wm in zip(w_R_m_lst, w_v_wm_arr)])
m_v_wm_arr_filt = np.array([w_R_m.T @ w_v_wm for w_R_m, w_v_wm in zip(w_R_m_lst, w_v_wm_arr_sagol)])
b_v_ob_arr = np.array([o_R_b.T @ o_v_ob for o_R_b, o_v_ob in zip(o_R_b_lst, o_v_ob_arr)])
if POVCDL:
    b_v_oc_arr = np.array([o_R_b.T @ o_v_oc for o_R_b, o_v_oc in zip(o_R_b_lst, o_v_oc_arr)])


#####################
# Trajectory alignment
pose_est = np.hstack([t_arr.reshape((N,1)), o_p_ob_arr, o_q_b_arr])
pose_gtr = np.hstack([t_arr.reshape((N,1)), w_p_wm_arr, w_q_m_arr])

# filter out end of traj where the robot solo causes strange jump in Mo-Cap position
# NPREV = 12
# t_arr = t_arr[:N-NPREV] 
# pose_est = pose_est[:N-NPREV,:]
# pose_gtr = pose_gtr[:N-NPREV,:]
# m_v_wm_arr_filt = m_v_wm_arr_filt[:N-NPREV,:]
# b_v_ob_arr = b_v_ob_arr[:N-NPREV,:]
# b_v_oc_arr = b_v_oc_arr[:N-NPREV,:]

res_folder = 'res_for_rpg_'+datetime.datetime.now().strftime("%y_%m_%d__%H_%M_%S") + '/'
os.makedirs(res_folder)
np.savetxt(res_folder+'stamped_traj_estimate.txt', pose_est, delimiter=' ')
np.savetxt(res_folder+'stamped_groundtruth.txt',   pose_gtr, delimiter=' ')
traj = rpg_traj.Trajectory(res_folder, align_type='se3', align_num_frames=1)  # settings ensured by eval_cfg.yaml

traj.compute_absolute_error()
# traj.compute_boxplot_distances()
# traj.compute_relative_errors()
# traj.compute_relative_error_at_subtraj_len()

# Create a continuous norm to map from data points to colors

# def plot_xy_traj(xy, cmap, fig, ax, cstart):
#     points = xy.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     norm = plt.Normalize(t_arr[0], t_arr[-1])
#     lc = LineCollection(segments, cmap=cmap, norm=norm)
#     # Set the values used for colormapping
#     lc.set_array(t_arr)
#     lc.set_linewidth(2)
#     line = ax.add_collection(lc)
#     # fig.colorbar(line, ax=ax)
#     plt.plot(xy[0,0], xy[0,1], cstart+'x')


# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# plot_xy_traj(traj.p_es_aligned[:,:2], 'autumn', fig, ax, 'b')
# plot_xy_traj(traj.p_gt[:,:2], 'winter', fig, ax, 'r')
# fig.tight_layout()

# xmin = traj.p_es_aligned[:,0].min() 
# xmax = traj.p_es_aligned[:,0].max() 
# ymin = traj.p_es_aligned[:,1].min() 
# ymax = traj.p_es_aligned[:,1].max()

# xmin = min(xmin, traj.p_gt[:,0].min()) 
# xmax = max(xmax, traj.p_gt[:,0].max()) 
# ymin = min(ymin, traj.p_gt[:,1].min()) 
# ymax = max(ymax, traj.p_gt[:,1].max())

# offx = 0.1*(xmax - xmin)
# offy = 0.1*(ymax - ymin)

# # line collections don't auto-scale the plot
# plt.xlim(xmin-offx, xmax+offx) 
# plt.ylim(ymin-offy, ymax+offy)
# plt.grid()





# Compute orientation as roll pitch yaw
R_es_aligned = [pin.Quaternion(q.reshape((4,1))).toRotationMatrix() for q in traj.q_es_aligned]
R_gt = [pin.Quaternion(q.reshape((4,1))).toRotationMatrix() for q in traj.q_gt]
rpy_es_aligned = np.array([pin.rpy.matrixToRpy(R) for R in R_es_aligned])
rpy_gt = np.array([pin.rpy.matrixToRpy(R) for R in R_gt])



# PLOT parameters
# FIGSIZE = (3.14,2.8)
FIGSIZE = (6,5)
GRID = True
# EXT = '.png'
EXT = '.pdf'
if PGF:
    EXT = '.pgf'



fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
fig.canvas.set_window_title('base_orientation_base_frame')
ylabels = ['Roll [rad]', 'Pitch [rad]', 'Yaw [rad]']
for i in range(3):
    axs[i].plot(t_arr, rpy_es_aligned[:,i], 'b', markersize=1, label='est')
    axs[i].plot(t_arr, rpy_gt[:,i], 'r', markersize=1, label='Mo-Cap')
    axs[i].set_ylabel(ylabels[i])
    axs[i].yaxis.set_label_position("right")
    axs[i].grid(GRID)
axs[2].set_xlabel('time [s]')
# axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
axs[0].legend()
fig.savefig(res_folder+'base_orientation_base_frame'+EXT)

fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
fig.canvas.set_window_title('base_velocity_base_frame')
ylabels = ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]']
for i in range(3):
    axs[i].plot(t_arr, b_v_ob_arr[:,i], 'b', markersize=1, label='est')
    axs[i].plot(t_arr, m_v_wm_arr_filt[:,i], 'r', markersize=1, label='Mo-Cap')
    axs[i].set_ylabel(ylabels[i])
    axs[i].yaxis.set_label_position("right")
    axs[i].grid(GRID)
axs[2].set_xlabel('time [s]')
# axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
axs[0].legend()
fig.savefig(res_folder+'base_velocity_base_frame'+EXT)

fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
fig.canvas.set_window_title('base_position')
ylabels = ['Px [m]', 'Py [m]', 'Pz [m]']
for i in range(3):
    axs[i].plot(t_arr, traj.p_es_aligned[:,i], 'b', markersize=1, label='est')
    axs[i].plot(t_arr, traj.p_gt[:,i], 'r', markersize=1, label='Mo-Cap')
    axs[i].set_ylabel(ylabels[i])
    axs[i].yaxis.set_label_position("right")
    axs[i].grid(GRID)
axs[2].set_xlabel('time [s]')
# axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
axs[0].legend()
fig.savefig(res_folder+'base_position'+EXT)


if POVCDL:
    fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
    fig.canvas.set_window_title('com_position')
    ylabels = ['Px [m]', 'Py [m]', 'Pz [m]']
    for i in range(3):
        axs[i].plot(t_arr, o_p_ob_arr[:,i], 'b', markersize=1, label='base')
        axs[i].plot(t_arr, o_p_oc_arr[:,i], 'r', markersize=1, label='com')
        axs[i].set_ylabel(ylabels[i])
        axs[i].yaxis.set_label_position("right")
        axs[i].grid(GRID)
    axs[2].set_xlabel('time [s]')
    # axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
    axs[0].legend()
    fig.savefig(res_folder+'com_position'+EXT)


    fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
    fig.canvas.set_window_title('com_position')
    ylabels = ['Px [m]', 'Py [m]', 'Pz [m]']
    for i in range(3):
        axs[i].plot(t_arr, o_p_ob_arr[:,i], 'b', markersize=1, label='base')
        axs[i].plot(t_arr, o_p_oc_arr[:,i], 'r', markersize=1, label='com')
        axs[i].plot(t_arr, o_p_oc_arr[:,i], 'r', markersize=1, label='com')
        axs[i].set_ylabel(ylabels[i])
        axs[i].yaxis.set_label_position("right")
        axs[i].grid(GRID)
    axs[2].set_xlabel('time [s]')
    # axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
    axs[0].legend()
    fig.savefig(res_folder+'com_position_mocap'+EXT)


    fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
    fig.canvas.set_window_title('com_velocity')
    ylabels = ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]']
    for i in range(3):
        axs[i].plot(t_arr, o_v_ob_arr[:,i], 'b', markersize=1, label='base')
        axs[i].plot(t_arr, o_v_oc_arr[:,i], 'r', markersize=1, label='com')
        axs[i].set_ylabel(ylabels[i])
        axs[i].yaxis.set_label_position("right")
        axs[i].grid(GRID)
    axs[2].set_xlabel('time [s]')
    # axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
    axs[0].legend()
    fig.savefig(res_folder+'com_velocity'+EXT)




print('Figure saved in ' + res_folder)

plt.show()