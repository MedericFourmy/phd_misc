import numpy as np
import pandas as pd
import pinocchio as pin
from scipy import optimize
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas, shortened_arr_dic
from calibration_costs import CostOffsetNew, CostFlexiNew, CostFlexiOffsetNew, CostFlexiOffsetFrictionNew, CostOffsetDistances, CostOffsetDistancesGtr

robot = load('solo12')

dt = 1e-3

# Plank dimensions (< 13/10/21)
# width = 0.30
# length = 0.38

# Plank dimensions (>= 13/10/21)
width = 0.293
# width = 0.294
# length = 0.389
length = 0.390
height = 0.017

# ω_p_ωl_arr = np.array([
#     [ length/2,         width/2, 0],
#     [ length/2+0.05,  -(width/2+0.05), 0],
#     [-length/2,       -(width/2+0.05), 0],
#     [ -length/2,       +width/2, 0],
# ])

ω_p_ωl_arr = np.array([
    [ length/2,         width/2, 0],
    [ length/2+0.05,  -(width/2+0.05), 0],
    [-length/2,       -width/2, 0],
    [ -length/2,       +width/2, 0],
])

# print(ω_p_ωl_arr)


# number of sample extracted
N_sub = 8000
# Number of time we run the estimation
N_monte = 1


# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/calib_mocap_planck_move_Kp3.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/solocontrol/src/quadruped-reactive-walking/scripts/selected_qa.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_calib_Kp3_clean_slow.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner.npz'

# IRI 2nd time
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_19_10_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_bis_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_ter_format.npz'

# HIGHER
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_kp2_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_kp1_format.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2_format.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2_vib02_format.npz'

# OFFSET only (Gulliver)
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_offset_calibration_format.npz'




m_M_b_init = pin.SE3.Identity()
m_M_b_init.translation = -np.array([0.1163, 0.0, 0.02])
# w_M_ω_init = pin.SE3.Identity()
# w_M_ω_init.translation[2] = height

w_M_ω_init = pin.SE3.Identity()
# w_M_ω_init.translation[0] = -0.355
# w_M_ω_init.translation[1] = 0.210
w_M_ω_init.translation[2] = 0.8
# w_M_ω_init.translation[2] = height





arr_dic = np.load(file_path)

# arr_dic = read_data_file_laas(file_path, dt)
# arr_dic = shortened_arr_dic(arr_dic, S=500, N=-500)


LEGS = ['FL', 'FR', 'HR', 'HL']
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

t_arr = arr_dic['t']
qa_arr = arr_dic['qa']
tau_arr = arr_dic['tau']
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']

PHASE = 6
t_arr = t_arr[PHASE::10]
qa_arr = qa_arr[PHASE::10]
tau_arr = tau_arr[PHASE::10]
w_p_wm_arr = w_p_wm_arr[PHASE::10]
w_q_m_arr = w_q_m_arr[PHASE::10]

N = len(t_arr)

r_lst = []
for i in range(N_monte):
    print(i,'/',N_monte)
    # subsample the trajectory for quicker tests
    # np.random.seed(0)
    select = np.random.choice(np.arange(N), N_sub, replace=False)
    select.sort()

    params = {
        'robot': robot,
        'qa_arr': qa_arr[select],
        'dqa_arr': qa_arr[select],
        'tau_arr': tau_arr[select],
        'w_p_wm_arr': w_p_wm_arr[select],
        'w_q_m_arr': w_q_m_arr[select],
        'm_M_b_init': m_M_b_init,
        'w_M_ω_init': w_M_ω_init,
        'ω_p_ωl_arr': ω_p_ωl_arr,
        'height': height,
        'cids': cids,
        'N': N_sub,
        'LEGS': LEGS,
    }

    # cost = CostOffsetDistancesGtr(params)
    # x0 = np.zeros(12)

    cost = CostOffsetDistances(params)
    x0 = np.zeros(12+6)

    # cost = CostOffsetNew(params)
    # x0 = np.zeros(12+6+6)

    # cost = CostFlexiNew(params)
    # x0 = np.zeros(12+6+6)

    # cost = CostFlexiOffsetNew(params)
    # x0 = np.zeros(12+12+6+6)
    # x0 = 0.1*np.random.random(12+12+6+6)

    # cost = CostFlexiOffsetFrictionNew(params)
    # x0 = np.zeros(12+12+12+6+6)

    r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', loss='huber', verbose=2)

    r_lst.append(r)


cost.print_n_plot_solutions(r_lst)
cost.plot_residuals(r_lst[-1])
cost.save_calib(r_lst[-1])


plt.show()