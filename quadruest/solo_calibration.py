import numpy as np
import pandas as pd
import pinocchio as pin
from scipy import optimize
import matplotlib.pyplot as plt
from example_robot_data import load
from data_readers import read_data_file_laas

"""
goal: calibration of joint angle offsets by assuming feet distances are constant in a trajectory

minimize cost(delta_q, {d_ij}}]) = sum( sum( || dist_ij(q_t + delta_q)^2 - d_ij^2 ) ||^2 )
"""


robot = load('solo12')

LEGS = ['FL', 'FR', 'HR', 'HL']
# contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_names = [leg+'_FOOT' for leg in LEGS]
cids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

combinations = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3),
]
comb_names = [(LEGS[c[0]], LEGS[c[1]]) for c in combinations]
comb_ids = [(cids[c[0]], cids[c[1]]) for c in combinations]
print(comb_ids)
# print(combinations)
# print(comb_names)
# print(comb_names)
N_comb = len(combinations)

dt = 1e-3
# file_path = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_36.npz'  # sin smaller
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_38.npz'  # sin traj
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_41.npz'  # manual short
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_03_12_2020_palette/data_2020_12_03_17_44.npz'  # manual longer
arr_dic = read_data_file_laas(file_path, dt)
dab_arr_gtr = np.array([
    0.327,
    0.522,
    0.405,
    0.403,
    0.523,
    0.331
])

# # test with simu
# from numpy import genfromtxt
# # file_path = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_q.dat'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_traj_q.dat'
# qa_arr_simu = pd.read_csv(file_path, sep=' ', index_col=0).to_numpy()
# delta_simu = np.ones(12)*0.05
# arr_dic = {
#     't': np.arange(len(qa_arr_simu))*dt,
#     'qa': qa_arr_simu + delta_simu
# }
# # compute real distances for simu
# q = robot.q0.copy()
# q[7:] = qa_arr_simu[0,:]
# robot.forwardKinematics(q)
# dab_arr_gtr = np.zeros(N_comb)
# for ic, (combid, comb) in enumerate(zip(comb_ids, combinations)):
#     pa = robot.framePlacement(q, combid[0], update_kinematics=False).translation
#     pb = robot.framePlacement(q, combid[1], update_kinematics=False).translation
#     dab_arr_gtr[ic] = np.linalg.norm(pb - pa)



# remove beginning and end of trajectory 
t_arr = arr_dic['t']
N = len(t_arr)
S = 500
E = N - 30
t_arr = t_arr[S:E]
qa_arr = arr_dic['qa'][S:E]
N = len(t_arr)


# subsample the trajectory for quicker tests
# np.random.seed(0)
print('Dataset size: ', N)
N_sub = 20000
print('Sample size: ', N_sub)
select = np.random.choice(np.arange(N), N_sub, replace=False)
qa_arr = qa_arr[select]

# compute size of the residual
N_res = N_sub * N_comb


class Cost:
    def __init__(self, robot, qa_arr):
        self.robot = robot
        self.qa_arr = qa_arr
        self.x_arr = []
        self.cost_arr = []

    def f(self, x):
        self.x_arr.append(x)
        res = np.zeros(N_res)
        q = self.robot.q0.copy()

        for k, qa in enumerate(self.qa_arr):
            q[7:] = qa + x[:12]
            self.robot.forwardKinematics(q)
            for ic, cid in enumerate(comb_ids):
                pa = self.robot.framePlacement(q, cid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, cid[1], update_kinematics=True).translation
                res[N_comb*k + ic] = np.linalg.norm(pb - pa) - x[12+ic]


        self.cost_arr.append(np.linalg.norm(x))

        return res
    
    def jac(self, x):

        J = np.zeros((N_sub*N_comb,18))

        q = self.robot.q0.copy()
        for k, qa in enumerate(self.qa_arr):
            q[7:] = qa + x[:12]
            self.robot.forwardKinematics(q)
            for ic, cid in enumerate(comb_ids):
                pa = self.robot.framePlacement(q, cid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, cid[1], update_kinematics=True).translation
                v = pb - pa
                self.robot.computeJointJacobians(q)
                self.robot.framesForwardKinematics(q)
                # take only the jacobian wrt. the actuated part
                oJa = robot.getFrameJacobian(cid[0], rf_frame=pin.LOCAL_WORLD_ALIGNED)[:3, 6:]
                oJb = robot.getFrameJacobian(cid[1], rf_frame=pin.LOCAL_WORLD_ALIGNED)[:3, 6:]
                
                # jacobian wrt. joint deltas
                J[N_comb*k + ic,:12] = v.T@(oJb - oJa) / np.linalg.norm(v)
                # jacobian wrt. combination distances 
                J[N_comb*k + ic,12+ic] = -1

        return J

def check_finite_difference(cost):
    x = np.random.random(18)
    eps = 1e-10

    Jan = cost.jac(x)

    Jfin = np.zeros(Jan.shape)

    for i in range(18):
        dx = np.zeros(18)
        dx[i] = eps
        y = cost.f(x)
        ydx = cost.f(x+dx)
        Jfin[:,i] = (ydx - y)/eps

    assert(np.allclose(Jan, Jfin, atol=1e-5))


        
        
cost = Cost(robot, qa_arr)
# check_finite_difference(cost)


# initialize distances with random configuration
q = robot.q0.copy()
q[7:] = qa_arr[np.random.choice(np.arange(N_sub), 1)[0], :]
robot.forwardKinematics(q)
dab_arr = np.zeros(N_comb)
for ic, (combid, comb) in enumerate(zip(comb_ids, combinations)):
    pa = robot.framePlacement(q, combid[0], update_kinematics=False).translation
    pb = robot.framePlacement(q, combid[1], update_kinematics=False).translation
    dab_arr[ic] = np.linalg.norm(pb - pa)


x0 = np.concatenate([np.zeros(12), dab_arr])
# x0 = np.concatenate([np.zeros(12), dab_arr_gtr])
# x0 = np.zeros(18)

# x0 = np.array([
#     -0.01059897,  0.10145532, -0.08407983, -0.06633118,  0.12592768, -0.10973976,
#        0.05596756, -0.11101342,  0.1350453,  -0.05010525, -0.15712572,  0.15532325,
#        0.33500665, 0.51593649, 0.39655017, 0.39228093, 0.52615419, 0.34615445])

# r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
# r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', verbose=2)
# r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', loss='huber', verbose=2)
r = optimize.least_squares(cost.f, x0, jac=cost.jac, method='trf', loss='huber', verbose=2)


print()
print('r.cost')
print(r.cost)
print()
print('deltas')
print(r.x[:12])
print()
print('distances optim')
print(r.x[12:])
print('distances gtr')
print(dab_arr_gtr)

# residuals RMSE
res = r.fun.reshape((N_sub, N_comb))
rmse_arr = np.sqrt(np.mean(res**2, axis=0))

with open('x_est.csv','a') as fd:
    fd.write(','.join(str(xi) for xi in r.x)+'\n')

with open('rmse_res.csv','a') as fd:
    fd.write(','.join(str(rmse) for rmse in rmse_arr)+'\n')
