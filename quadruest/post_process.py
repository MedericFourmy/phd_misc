import sys
import numpy as np
from scipy import signal
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# uses https://github.com/uzh-rpg/rpg_trajectory_evaluation.git
sys.path.append('/home/mfourmy/Documents/Phd_LAAS/installations/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation') 
import trajectory as rpg_traj


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
data_file = 'data_2020_10_09_19_03.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_03.npz_bak'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_05.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_06.npz'

# Keys:
res_arr_dic = np.load(DATA_FOLDER_RESULTS+data_file)

dt = 1e-3
t_arr = res_arr_dic['t']
N = len(t_arr)
# GT
w_p_wm_arr = res_arr_dic['w_p_wm']
w_q_m_arr = res_arr_dic['w_q_m']
w_v_wm_arr = res_arr_dic['w_v_wm']
m_v_wm_arr = res_arr_dic['m_v_wm']
i_omg_oi_arr = res_arr_dic['i_omg_oi']

o_R_b_arr = res_arr_dic['o_R_b']
o_q_b_arr = res_arr_dic['o_q_b']
o_p_ob_kf_arr = res_arr_dic['o_p_ob_kf']
o_v_ob_kf_arr = res_arr_dic['o_v_ob_kf']
o_p_ob_cf_arr = res_arr_dic['o_p_ob_cf']
o_v_ob_cf_arr = res_arr_dic['o_v_ob_cf']


pose_est = np.hstack([t_arr.reshape((N,1)), o_p_ob_kf_arr, o_q_b_arr])
pose_gtr = np.hstack([t_arr.reshape((N,1)), w_p_wm_arr, w_q_m_arr])

np.savetxt('stamped_traj_estimate.txt', pose_est, delimiter=' ')
np.savetxt('stamped_groundtruth.txt',   pose_gtr, delimiter=' ')

traj = rpg_traj.Trajectory('.')  # settings ensured by eval_cfg.yaml
# traj.compute_absolute_error()
# traj.compute_boxplot_distances()
# traj.compute_relative_errors()
# traj.compute_relative_error_at_subtraj_len()

# Create a continuous norm to map from data points to colors
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

def plot_xy_traj(xy, cmap, fig, ax, cstart):
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t_arr[0], t_arr[-1])
    # in fact 0 -> white for most colormaps, bad to visualize -> add offset
    # [A, B] --> [a, b]
    # (val - A)*(b-a)/(B-A) + a
    # [0,1] --> [offset, 1]
    # offset = 0.4
    # norm = (norm-offset)*(1-offset) + offset
    # print(norm.min())
    # print(norm.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(t_arr)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)
    plt.plot(xy[0,0], xy[0,1], cstart+'x')

plot_xy_traj(traj.p_es_aligned[:,:2], 'autumn', fig, ax, 'b')
plot_xy_traj(traj.p_gt[:,:2], 'winter', fig, ax, 'r')
fig.tight_layout()

xmin = traj.p_es_aligned[:,0].min() 
xmax = traj.p_es_aligned[:,0].max() 
ymin = traj.p_es_aligned[:,1].min() 
ymax = traj.p_es_aligned[:,1].max()

xmin = min(xmin, traj.p_gt[:,0].min()) 
xmax = max(xmax, traj.p_gt[:,0].max()) 
ymin = min(ymin, traj.p_gt[:,1].min()) 
ymax = max(ymax, traj.p_gt[:,1].max())

offx = 0.1*(xmax - xmin)
offy = 0.1*(ymax - ymin)

# line collections don't auto-scale the plot
plt.xlim(xmin-offx, xmax+offx) 
plt.ylim(ymin-offy, ymax+offy)


NB_OVER = 5

# compute filterd mocap velocity
w_v_wm_filtered_arr = signal.savgol_filter(w_p_wm_arr, window_length=19, polyorder=5, deriv=1, axis=0, delta=dt, mode='mirror')
w_p_wm_arr_sub = w_p_wm_arr[::NB_OVER]
w_p_wm_arr_over = w_p_wm_arr_sub.repeat(NB_OVER, axis=0) 
w_v_wm_arr_sub = signal.savgol_filter(w_p_wm_arr_sub, window_length=19, polyorder=5, deriv=1, axis=0, delta=NB_OVER*dt, mode='mirror')
w_v_wm_arr_over = w_v_wm_arr_sub.repeat(NB_OVER, axis=0) 

plt.figure('mocap velocity')
plt.subplot(3,1,1)
plt.plot(t_arr, w_v_wm_arr[:,0], 'r.', markersize=1, label='vx numdiff')
plt.plot(t_arr, w_v_wm_filtered_arr[:,0], 'b.', markersize=1, label='vx filtered')
plt.plot(t_arr, w_v_wm_arr_over[:,0], 'k.', markersize=1, label='vx filtered')
plt.subplot(3,1,2)
plt.plot(t_arr, w_v_wm_arr[:,1], 'r.', markersize=1, label='vy numdiff')
plt.plot(t_arr, w_v_wm_filtered_arr[:,1], 'b.', markersize=1, label='vy filtered')
plt.plot(t_arr, w_v_wm_arr_over[:,1], 'k.', markersize=1, label='vx filtered better')
plt.subplot(3,1,3)
plt.plot(t_arr, w_v_wm_arr[:,2], 'r.', markersize=1, label='vz numdiff')
plt.plot(t_arr, w_v_wm_filtered_arr[:,2], 'b.', markersize=1, label='vz filtered')
plt.plot(t_arr, w_v_wm_arr_over[:,2], 'k.', markersize=1, label='vz filtered better')
plt.legend()


# plt.figure('mocap position')
# plt.plot(t_arr[:19], w_p_wm_arr[:19,0], 'r.')
# plt.plot(t_arr[:19], w_p_wm_arr_over[:19,0], 'b.')


plt.show()