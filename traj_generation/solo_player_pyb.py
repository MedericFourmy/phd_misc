import time
import numpy as np
import pandas as pd
import pinocchio as pin
import matplotlib.pyplot as plt
import pybullet as pyb  # Pybullet server
from example_robot_data import load
from simulator_pybullet import SimulatorPybullet


def compute_control_torque(qa_d, va_d, tau_d, qa_mes, va_mes):
    Kp = 6
    Kd = 0.3
    err_qa = qa_d - qa_mes
    err_va = va_d - va_mes

    # tau = Kp*err_qa                      # elastic
    # tau = Kp*err_qa + Kd*err_va            # good
    tau = tau_d + Kp*err_qa + Kd*err_va  # torque makes the feet slide
    return tau

        
DT = 0.001
SLEEP = False
GUION = True


robot = load('solo12')


# From ContactSequence file
traj_dir = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
traj_file = 'solo_stamping.npz'

arr = np.load(traj_dir+traj_file)

q_arr = arr['q']
v_arr = arr['v']
tau_arr = arr['tau_ff']

q_init = q_arr[0,:]
q_init[2] += 0.00132   # Optimal to get the lowest BUMP for point feet of ~0 radius

URDF_NAME = 'solo12.urdf'
LEGS = ['FL', 'FR', 'HL', 'HR']
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]  # # same thing as XX_FOOT but contained in pybullet -> solo12

controlled_joint_names = []
for leg in LEGS:
    controlled_joint_names += [leg + '_HAA', leg + '_HFE', leg + '_KFE']

sim = SimulatorPybullet(URDF_NAME, DT, q_init, 12, robot, controlled_joint_names, contact_frame_names, guion=GUION, gravity=[0, 0.0, -9.81])

N = len(q_arr)-1

# Center the camera on the current position of the robot
pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
                            cameraTargetPosition=[q_init[0], q_init[1], 0.0])


for i in range(N):
    t1 = time.time()

    # STORE
    # Get position/orientation of the base and angular position of actuators
    sim.retrieve_pyb_data()
    q = sim.q  # w_p_b, w_q_b, qa
    v = sim.v  # nu_b, va

    # torque to be exactly applied on the robot pyb model joints
    tau = compute_control_torque(q_arr[i,7:], v_arr[i,6:], tau_arr[i,:], q[7:], v[6:])


    # Set control torque for all joints
    pyb.setJointMotorControlArray(sim.robotId, sim.bullet_joint_ids,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # Compute one step of simulation
    pyb.stepSimulation()
  
    # Wait a bit
    delay = time.time() - t1
    if SLEEP:
        time.sleep(delay)


plt.show()