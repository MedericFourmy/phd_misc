import time
import numpy as np
import pandas as pd
import pinocchio as pin
import matplotlib.pyplot as plt
import pybullet as pyb  # Pybullet server

from simulator_pybullet import SimulatorPybullet


path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'
srdf = path + '/srdf/solo.srdf'
reference_config_q_name = 'standing'

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
pin.loadReferenceConfigurations(robot.model, srdf, False)
q0 = robot.model.referenceConfigurations[reference_config_q_name]
# robot.initViewer(loadModel=True)  

file_qtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_q.dat'
file_vtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_v.dat'
file_tautraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_tau.dat'
df_q = pd.read_csv(file_qtraj, sep=' ', header=None, index_col=0)
df_v = pd.read_csv(file_vtraj, sep=' ', header=None, index_col=0)
df_tau = pd.read_csv(file_tautraj, sep=' ', header=None, index_col=0)

q_arr = df_q.to_numpy()
v_arr = df_v.to_numpy()
tau_arr = df_tau.to_numpy()

q_init = robot.model.referenceConfigurations['standing']
sim = SimulatorPybullet(dt=0.001, q_init=q_init, nqa=12, gravity=[0, 0.0, -9.81])

# q_init[2] = 1
# sim = SimulatorPybullet(dt=0.001, q_init=q_init, nqa=12, gravity=[0, 0.0, 0])

N = len(df_q)

delays = []
for i in range(N):
    t1 = time.time()

    # Get position/orientation of the base and angular position of actuators
    sim.retrieve_pyb_data()

    # Center the camera on the current position of the robot
    pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-50, cameraPitch=-39.9,
                                cameraTargetPosition=[sim.qmes12[0, 0], sim.qmes12[1, 0], 0.0])

    # Set control torque for all joints
    qa_d = q_arr[i,:]
    va_d = v_arr[i,:]
    tau_d = tau_arr[i,:]

    qa_mes = sim.qmes12[7:,0]
    va_mes = sim.vmes12[6:,0]

    Kp = 7
    Kd = 0.5
    # Kd = 2*np.sqrt(Kp)  # NOPE
    err_qa = qa_d - qa_mes
    err_va = va_d - va_mes

    # tau = Kp*err_qa                      # elastic
    tau = Kp*err_qa + Kd*err_va            # good
    # tau = tau_d + Kp*err_qa + Kd*err_va  # torque makes the feet slide

    pyb.setJointMotorControlArray(sim.robotId, sim.revoluteJointIndices,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # Compute one step of simulation
    pyb.stepSimulation()

    # Wait a bit
    delay = time.time() - t1
    delays.append(delay)
    time.sleep(0.001)

plt.figure()
plt.plot(np.arange(N), np.array(delays))
plt.grid()
plt.show()