import numpy as np
import pandas as pd
import pinocchio as pin
import time
from example_robot_data import loadSolo, loadTalos

path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'
srdf = path + '/srdf/solo.srdf'
reference_config_q_name = 'standing'

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
pin.loadReferenceConfigurations(robot.model, srdf, False)
q0 = robot.model.referenceConfigurations[reference_config_q_name]
robot.initViewer(loadModel=True)  

file_qtraj = '/home/mfourmy/Documents/Phd_LAAS/general_phd_repo/traj_generation/temp_traj/solo_stamping_q_with_pose.dat'
# file_qtraj = '/home/mfourmy/Documents/Phd_LAAS/general_phd_repo/traj_generation/temp_traj/solo_stamping_q.dat'
df_q = pd.read_csv(file_qtraj, sep=' ', header=None)
q_arr = df_q.to_numpy()[:,1:]  # remove index and switch to array


N = q_arr.shape[0]

if q_arr.shape[1] == robot.nq-7:
    # if base pose not in file, put the robot at default pose
    pose_arr = q0[:7] * np.ones((N,1))
    q_arr = np.hstack([pose_arr, q_arr])
elif not q_arr.shape[1] == robot.nq:
    raise Exception('Wrong nb of column in dat file, check your robot model')
    


dt = 1e-3
display_every = 5
display_dt = display_every * dt

robot.q0

for i in range(N):
    if (i % display_every) == 0:
        time_start = time.time()
        q = q_arr[i,:]
        robot.display(q)

        time_spent = time.time() - time_start
        if(time_spent < display_dt): time.sleep(display_dt-time_spent)


