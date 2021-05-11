import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import load

robot_plan = load('solo12')
robot_real = load('solo12')
robot_plan.initViewer(loadModel=True, sceneName='world/plan')
robot_real.initViewer(loadModel=True, sceneName='world/real')

gv = robot_plan.viewer.gui
window_id = 'python-pinocchio'
# gv.addFloor('hpp-gui/floor')

for node in gv.getNodeList():
    if '/plan' in node:
        gv.setColor(node,[0,1,0,0.5])


file_qtraj_plan = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_sin_smaller_q.dat'
file_qtraj_real = '/home/mfourmy/Documents/Phd_LAAS/solo-estimation/data/Point_feet_27_11_20/data_2020_11_27_14_46.npz'

df_qa_plan = pd.read_csv(file_qtraj_plan, sep=' ', header=None)
qa_plan_arr = df_qa_plan.to_numpy()[:,1:]  # remove index and switch to array

arr_dic = np.load(file_qtraj_real)
qa_real_arr = arr_dic['q_mes']
w_p_wm_arr = arr_dic['mocapPosition']
w_q_m_arr = arr_dic['mocapOrientationQuat']

N = qa_real_arr.shape[0]
Nbis = qa_real_arr.shape[0]

print(N)
print(Nbis)


dt = 1e-3
display_every = 5
display_dt = display_every * dt

for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        qa_plan = qa_plan_arr[i,:]
        qa_real = qa_real_arr[i,:]
        w_p_wm = w_p_wm_arr[i,:]
        w_q_m = w_q_m_arr[i,:]
        q_plan = np.concatenate([w_p_wm, w_q_m, qa_plan])
        q_real = np.concatenate([w_p_wm, w_q_m, qa_real])
        robot_plan.display(q_plan)
        robot_real.display(q_real)

        time_spent = time.time() - time_start
        if(time_spent < display_dt): time.sleep(display_dt-time_spent)


