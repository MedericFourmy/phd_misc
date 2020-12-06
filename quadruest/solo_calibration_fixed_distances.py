import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas

"""
goal: calibration of joint angle offsets by measuring constant distances between feet

minimize cost(delta_q, {d_ij}}]) = sum( sum( || dist_ij(q_t + delta_q)^2 - d_ij^2 ) ||^2 )

Notes:
- results vary quite a lot depending on the chosen samples
- shoulder joints seem to vary much more (5e-3) than other joints ()
"""


dt = 1e-3
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_36.npz'  # sin smaller
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_38.npz'  # sin traj
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_41.npz'  # manual short
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_44.npz'  # manual longer
# arr_dic = read_data_file_laas(file_path, dt)

# test with simu
from numpy import genfromtxt
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_q.dat'
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_traj_q.dat'
qa_arr_simu = genfromtxt(file_path, delimiter=',')
arr_dic = {
    't': np.arange(len(arr))*dt,
    'qa': qa_arr_simu
}

VIEW = False
robot = load('solo12')

LEGS = ['FL', 'FR', 'HR', 'HL']
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
combids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

# remove beginning and end of trajectory 
t_arr = arr_dic['t']
N = len(t_arr)
S = 500 
E = N - 30
t_arr = t_arr[S:E]
qa_arr = arr_dic['qa'][S:E]
N = len(t_arr)

combinations = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3),
]
comb_names = [(LEGS[c[0]], LEGS[c[1]]) for c in combinations]
comb_ids = [(combids[c[0]], combids[c[1]]) for c in combinations]
# print(combinations)
# print(comb_names)
N_comb = len(combinations)

dists = {
    (0,1): 0.327,
    (0,2): 0.405,
    (0,3): 0.522,
    (1,2): 0.523,
    (1,3): 0.403,
    (2,3): 0.331,
}

# subsample the trajectory for quicker tests
N_sub = 5000
select = np.random.choice(np.arange(N), N_sub, replace=False)
qa_arr = qa_arr[select]

# compute size of the residual
N_res = N_sub * (12+N_comb)


class Cost:
    def __init__(self, robot, qa_arr):
        self.robot = robot
        self.qa_arr = qa_arr
        self.x_arr = []
        self.cost_arr = []

    def f(self, x):
        self.x_arr.append(x)
        delta_qa = x
        res = np.zeros(N_res)
        q = self.robot.q0.copy()

        for it, qa in enumerate(self.qa_arr):
            q[7:] = qa + delta_qa
            self.robot.forwardKinematics(q)
            for ic, (combid, comb) in enumerate(zip(comb_ids, combinations)):
                pa = self.robot.framePlacement(q, combid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, combid[1], update_kinematics=True).translation
                res[N_comb*it + ic] = np.linalg.norm(pb - pa) - dists[comb]


        self.cost_arr.append(np.linalg.norm(x))

        return res
        
        
cost = Cost(robot, qa_arr)


x0 = np.zeros(12)
r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)

print()
print('r.x')
print(r.x)
print()
print('r.cost')
print(r.cost)
print()

with open('delta_qa_est.csv','a') as fd:
    fd.write(','.join(str(qi) for qi in r.x)+'\n')

# print('r.fun')
# print(r.fun)