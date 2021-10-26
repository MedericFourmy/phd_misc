import numpy as np
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas, shortened_arr_dic

dt = 1e-3
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_36.npz'  # sin smaller
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_38.npz'  # sin traj
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_41.npz'  # manual short
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_44.npz'  # manual longer

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Calibration_Manuelle_09_12_2020/data_2020_12_09_14_46.npz'  # manual NO CONTROL
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Calibration_Manuelle_09_12_2020/data_2020_12_09_14_48.npz'  # manual NO CONTROL


# Data october 2021 (Joan's stay)
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_38_30__Kp3.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_38_30__Kp7.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_43_35__Kp7.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_still.npz'
file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_move_Kp6.npz'




arr_dic = read_data_file_laas(file_path, dt)
arr_dic = shortened_arr_dic(arr_dic, S=5000, N=-5000)

VIEW = False
robot = load('solo12')

LEGS = ['FL', 'FR', 'HR', 'HL']
# contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
contact_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

t_arr = arr_dic['t']
N = len(t_arr)
t_arr = t_arr
w_p_wm_arr = arr_dic['w_p_wm']
w_R_m_arr = arr_dic['w_R_m']
w_q_m_arr = arr_dic['w_q_m']
qa_arr = arr_dic['qa']
tau_arr = arr_dic['tau']
N = len(t_arr)

# qa_arr[:,0] += 1

delta_qa = np.zeros(12)


alpha = np.array([-8.07548808e-02, -2.62813516e-04,  3.31295563e-02, 
                  -2.83810418e-01, -9.76947822e-02, -1.32374969e-02, 
                  -7.91086092e-02,  5.90830806e-03, -8.94592668e-03, 
                  -9.06160815e-02,  3.36756692e-02, -3.19454471e-02,])

combinations = [
    (0,1),
    (1,2),
    (2,3),
    (0,3),
    # (0,2),
    # (1,3),
]
nbc = len(combinations)

for comb in combinations:
    print(LEGS[comb[0]], LEGS[comb[1]])

o_p_ol_arr_lst = [np.zeros((N,3)) for _ in range(nbc)]
o_p_ol_corr_arr_lst = [np.zeros((N,3)) for _ in range(nbc)]
distances = {comb: np.zeros((N)) for comb in combinations}
distances_corr = {comb: np.zeros((N)) for comb in combinations}
for i in range(N):
    w_p_wm = w_p_wm_arr[i,:]
    w_q_m = w_q_m_arr[i,:]
    # qa = qa_arr[i,:]
    qa = qa_arr[i,:] + delta_qa + alpha*tau_arr[i,:]
    q = np.concatenate([w_p_wm, w_q_m, qa_arr[i,:]])
    q_corr = np.concatenate([w_p_wm, w_q_m, qa])

    if VIEW:
        robot.display(q)
    
    robot.forwardKinematics(q)
    for l in range(nbc):
        o_p_ol_arr_lst[l][i,:] = robot.framePlacement(q, contact_ids[l], update_kinematics=False).translation
    for comb in combinations:
        distances[comb][i] = np.linalg.norm(o_p_ol_arr_lst[comb[0]][i,:] - o_p_ol_arr_lst[comb[1]][i,:]) 

    # same but with corrected data
    robot.forwardKinematics(q_corr)
    for l in range(len(combinations)):
        o_p_ol_corr_arr_lst[l][i,:] = robot.framePlacement(q_corr, contact_ids[l], update_kinematics=False).translation
    for comb in combinations:
        distances_corr[comb][i] = np.linalg.norm(o_p_ol_corr_arr_lst[comb[0]][i,:] - o_p_ol_corr_arr_lst[comb[1]][i,:]) 

# # legs_to_plot = [2,3]  # bad on y!
# legs_to_plot = [0,1]  # slip on FR y
# plt.figure('Feet positions')
# for i in range(3):
#     plt.subplot(3,1,1+i)
#     for l in legs_to_plot:
#         plt.plot(t_arr, o_p_ol_arr_lst[l][:,i], label=LEGS[l])
#     plt.legend()

plt.figure('qmes')
plt.plot(t_arr, qa_arr)


plt.figure('Feet distances')
plt.title('Feet distances')
colors = 'bgrcmy'
dgtr = [0.354, 0.43, 0.35, 0.38]
for i, comb in enumerate(combinations):
    plt.plot(t_arr, distances[comb], colors[i], label='{}/{}'.format(LEGS[comb[0]], LEGS[comb[1]]), alpha=0.3)
    plt.plot(t_arr, distances_corr[comb], colors[i]+'--', label='{}/{}_corr'.format(LEGS[comb[0]], LEGS[comb[1]]))
    plt.hlines(dgtr[i], t_arr[0], t_arr[-1], colors[i])
    plt.legend()


plt.show()