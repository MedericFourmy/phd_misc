import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import loadSolo, loadTalos

VIEW_ONLY_THE_FIRST = None
SPEED = 1  # wait(dt*SPEED)
# VIEW_ONLY_THE_FIRST = 5000

# just to visualize point feets
# URDF_NAME = 'solo12_pointed_feet.urdf'
# path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
# urdf = path + '/robots/' + URDF_NAME
# srdf = path + '/srdf/solo.srdf'
# robot_mocap = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())

robot_est = loadSolo(solo=False)  # for the 12
robot_mocap = loadSolo(solo=False)  # for the 12
robot_est.initViewer(loadModel=True, sceneName='world/est')
robot_mocap.initViewer(loadModel=True, sceneName='world/mocap')

gv = robot_mocap.viewer.gui
window_id = 'python-pinocchio'
gv.addFloor('hpp-gui/floor')
gv.addSphere('hpp-gui/com', 0.02, [1,0,0,1])

# gv.setFloatProperty('world/mocap', 'Alpha', 0.5)
# color robots
for node in gv.getNodeList():
    if '/est' in node:
        gv.setColor(node,[0,1,0,0.5])
    # if '/mocap' in node:
    #     gv.setColor(node,[0,1,0,1])

# add a football
# for a scale of 0.2, diameter of the whole solo length ~ 0.6m
# so for scale of 0.1, rayon of 0.15m
r = 0.02
gv.createGroup('hpp-gui/ball')
for i in range(1,5):
    # gv.deleteNode('world/ball/w%d'%i,True)
    # gv.deleteNode('world/ball/b%d'%i,True)
    gv.addSphere( 'hpp-gui/ball/w%d'%i,r,[1,1,1,1]   )   
    gv.addSphere( 'hpp-gui/ball/b%d'%i,r,[0,0,0,1])   
eps = 1e-3
gv.applyConfiguration('hpp-gui/ball/w1',[ eps, eps, eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/w2',[-eps,-eps, eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/b1',[-eps, eps, eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/b2',[ eps,-eps, eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/b3',[ eps, eps,-eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/b4',[-eps,-eps,-eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/w3',[-eps, eps,-eps ,0,0,0,1])
gv.applyConfiguration('hpp-gui/ball/w4',[ eps,-eps,-eps ,0,0,0,1])



# cam_transform = [
#  2.089702606201172,
#  0.303322434425354,
#  0.9693921208381653,
#  0.30712175369262695,
#  0.47138550877571106,
#  0.6858270764350891,
#  0.4616418182849884]  # sinXYZ

cam_transform = [
 2.224198818206787,
 0.17640361189842224,
 0.7717007994651794,
 0.3369377553462982,
 0.47232457995414734,
 0.6629767417907715,
 0.47312188148498535]  # stamping



gv.setCameraTransform(window_id, cam_transform)

file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out_post.npz'

arr_dic = np.load(file_path)
t_arr = arr_dic['t']
N = len(t_arr)
a_p_ab_arr = arr_dic['a_p_ab']
w_p_wm_arr = arr_dic['w_p_wm']
a_q_b_arr = arr_dic['a_q_b']
qa_arr = arr_dic['qa']
a_p_ac_arr = arr_dic['a_p_ac']
a_L_arr = arr_dic['a_L']


qest_arr = np.hstack([a_p_ab_arr, a_q_b_arr, qa_arr])
qmocap_arr = np.hstack([w_p_wm_arr, a_q_b_arr, qa_arr])

dt = 1e-3
display_every = 5
display_dt = SPEED*(display_every * dt)

if VIEW_ONLY_THE_FIRST is not None:
    N = VIEW_ONLY_THE_FIRST

m = 4*2.5  # solo mass
Iinv =  1/((2/5)*m*r**2) * np.eye(3)
R_mom = np.eye(3)
for i in range(N):
    # aRb = pin.Quaternion(a_q_b_arr[i,:].reshape((4,1))).toRotationMatrix()
    a_L = a_L_arr[i,:]
    a_omg = Iinv @ a_L
    R_mom = pin.exp3(a_omg*dt)@R_mom
    if (i % display_every) == 0:
        time_start = time.time()
        qe = qest_arr[i,:]
        qm = qmocap_arr[i,:]
        c = a_p_ac_arr[i,:]
        robot_est.display(qe)
        robot_mocap.display(qm)

        
        # move the momentum ball and center of mass
        q_mom_lst = pin.Quaternion(R_mom).coeffs().tolist()
        gv.applyConfiguration('hpp-gui/ball', [c[0] ,c[1], 0.3]+q_mom_lst)
        gv.applyConfiguration('hpp-gui/com', [c[0] ,c[1], 0, 0,0,0,1 ])
        gv.refresh()

        time_spent = time.time() - time_start
        if(time_spent < display_dt): time.sleep(display_dt-time_spent)


