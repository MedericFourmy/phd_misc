import numpy as np
import pandas as pd
import pinocchio as pin
from example_robot_data import load

robot = load('solo12')
qa0 = robot.model.referenceConfigurations['standing'][7:]

traj_name = 'solo_sin_rots_lowdz'
directory = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'

df_q = pd.read_csv(directory+traj_name+'_q.dat', sep=' ', index_col=0)
df_v = pd.read_csv(directory+traj_name+'_v.dat', sep=' ', index_col=0)
df_tau = pd.read_csv(directory+traj_name+'_tau.dat', sep=' ', index_col=0)

dt = 1e-3
DT = 20  # static robot for __ seconds
N = int(DT/dt)

def concat(arr0, df):
    return pd.DataFrame(np.vstack([arr0, df.to_numpy()]))

# things to append at the beginning
arr_q_0 = np.array([qa0 for _ in range(N)])
arr_v_0 = np.zeros((N,12))
arr_tau_0 = np.zeros((N,12))

df_q_sta = concat(arr_q_0, df_q)
df_v_sta = concat(arr_v_0, df_v)
df_tau_sta = concat(arr_tau_0, df_tau)

df_q_sta.to_csv(directory+traj_name+'_q_sta.dat', header=None, sep=' ')
df_v_sta.to_csv(directory+traj_name+'_v_sta.dat', header=None, sep=' ')
df_tau_sta.to_csv(directory+traj_name+'_tau_sta.dat', header=None, sep=' ')

print('Saved ', directory+traj_name+'_*_sta.dat', 'files')