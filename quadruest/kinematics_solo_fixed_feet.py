import numpy as np
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas

dt = 1e-3
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_36.npz'  # sin smaller
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_38.npz'  # sin traj
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_41.npz'  # manual short
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_44.npz'  # manual longer

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Calibration_Manuelle_09_12_2020/data_2020_12_09_14_46.npz'  # manual NO CONTROL
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Calibration_Manuelle_09_12_2020/data_2020_12_09_14_48.npz'  # manual NO CONTROL
arr_dic = read_data_file_laas(file_path, dt)

VIEW = False
robot = load('solo12')

LEGS = ['FL', 'FR', 'HR', 'HL']
# contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
contact_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

t_arr = arr_dic['t']
N = len(t_arr)
S = 500 
E = N - 30
t_arr = t_arr[S:E]
w_p_wm_arr = arr_dic['w_p_wm'][S:E]
w_R_m_arr = arr_dic['w_R_m'][S:E]
w_q_m_arr = arr_dic['w_q_m'][S:E]
qa_arr = arr_dic['qa'][S:E]
N = len(t_arr)

qa_arr[:,0] += 1

delta_qa = np.array([
    -0.01122508,  0.00826484, -0.04627015,  0.02105599, -0.01103931, -0.00173937,
     0.0111513 , -0.03019726,  0.07182457,  0.02774821,  0.01398398,  0.00784015,
     ])

combinations = [
    # (0,1),
    # (0,2),
    (0,3),
    # (1,2),
    # (1,3),
    # (2,3),
]

for comb in combinations:
    print(LEGS[comb[0]], LEGS[comb[1]])

o_p_ol_arr_lst = [np.zeros((N,3)) for _ in range(4)]
o_p_ol_corr_arr_lst = [np.zeros((N,3)) for _ in range(4)]
distances = {comb: np.zeros((N)) for comb in combinations}
distances_corr = {comb: np.zeros((N)) for comb in combinations}
for i in range(N):
    w_p_wm = w_p_wm_arr[i,:]
    w_q_m = w_q_m_arr[i,:]
    # qa = qa_arr[i,:]
    qa = qa_arr[i,:] + delta_qa
    q = np.concatenate([w_p_wm, w_q_m, qa_arr[i,:]])
    q_corr = np.concatenate([w_p_wm, w_q_m, qa])

    if VIEW:
        robot.display(q)
    
    robot.forwardKinematics(q)
    for l in range(4):
        o_p_ol_arr_lst[l][i,:] = robot.framePlacement(q, contact_ids[l], update_kinematics=False).translation
    for comb in combinations:
        distances[comb][i] = np.linalg.norm(o_p_ol_arr_lst[comb[0]][i,:] - o_p_ol_arr_lst[comb[1]][i,:]) 

    # same but with corrected data
    robot.forwardKinematics(q_corr)
    for l in range(4):
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


plt.figure('Feet distances')
plt.title('Feet distances')
colors = 'bgrcmy'
for i, comb in enumerate(combinations):
    plt.plot(t_arr, distances[comb], colors[i], label='{}/{}'.format(LEGS[comb[0]], LEGS[comb[1]]))
    plt.plot(t_arr, distances_corr[comb], colors[i]+'--', label='{}/{}_corr'.format(LEGS[comb[0]], LEGS[comb[1]]))
    plt.legend()

plt.figure('qmes')
plt.plot(t_arr, qa_arr)

plt.show()