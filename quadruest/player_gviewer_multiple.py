import time
import numpy as np
from example_robot_data import load

dt = 1e-3
SLEEP = True

# OUT OF ESTIMATION WOLF
DIRECTORY = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/centroidalkin/figs/window_experiments/'

traj = 'solo_trot_round_feet_format_feet_radius/out_viz1.npz'

arr_dic = np.load(DIRECTORY+traj)
w_pose_wm_est_arr = arr_dic['w_pose_wm']
w_pose_wm_fbk_arr = arr_dic['w_pose_wm_fbk']
w_pose_wm_gtr_arr = arr_dic['w_pose_wm_gtr']
w_pose_wm_cpf_arr = arr_dic['w_pose_wm_cpf']
qa_arr = arr_dic['qa']


robot_est = load('solo12')
robot_fbk = load('solo12')
robot_gtr = load('solo12')
robot_cpf = load('solo12')



robot_est.initViewer(loadModel=True, sceneName='world/est')
robot_fbk.initViewer(loadModel=True, sceneName='world/fbk')
robot_gtr.initViewer(loadModel=True, sceneName='world/gtr')
robot_cpf.initViewer(loadModel=True, sceneName='world/cpf')



gv = robot_est.viewer.gui
window_id = 'python-pinocchio'
gv.addFloor('hpp-gui/floor')
# gv.setLightingMode('hpp-gui/floor', 'OFF')
# gv.setColor('hpp-gui/floor', [1,1,1,1])
gv.setCameraTransform(window_id,
[-0.34668663144111633,
 -4.102753162384033,
 2.2448043823242188,
 0.5750080347061157,
 0.0273003987967968,
 0.012223783880472183,
 0.8176007866859436])

for node in gv.getNodeList():
    if '/est' in node:
        gv.setColor(node,[1,0,0,0.5])
    if '/fbk' in node:
        gv.setColor(node,[0,1,0,0.5])
    if '/cpf' in node:
        gv.setColor(node,[0,0,1,0.5])


N = qa_arr.shape[0]

display_every = 20
display_dt = display_every * dt


for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        robot_est.display(np.concatenate([w_pose_wm_est_arr[i,:], qa_arr[i,::]]))
        robot_fbk.display(np.concatenate([w_pose_wm_fbk_arr[i,:], qa_arr[i,::]]))
        robot_gtr.display(np.concatenate([w_pose_wm_gtr_arr[i,:], qa_arr[i,::]]))
        robot_cpf.display(np.concatenate([w_pose_wm_cpf_arr[i,:], qa_arr[i,::]]))
        gv.refresh()

        time_spent = time.time() - time_start
        if(SLEEP and time_spent < display_dt): time.sleep(display_dt-time_spent)


