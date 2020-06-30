#!/usr/bin/env python
# coding: utf-8

import sys
import time
import numpy as np
import pinocchio as pin
from example_robot_data import loadANYmal, loadTalos
import curves
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt; plt.ion()

QUADRUPED = False
DISPLAY_TRAJ = False
WORLD_FRAME = False
# examples_dir = ''
# file_name = 'anymal_walk_WB.cs'


if QUADRUPED:
    DIMF = 3  # point feet -> 3D force
    robot = loadANYmal()
    examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/multicontact-api-master-examples/examples/quadrupeds/'
    file_name = 'anymal_walk_WB_smoother.cs'
else:
    DIMF = 6  # humanoid feet -> 6D wrench
    robot = loadTalos()
    # examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/tsid_gen/'
    # file_name = 'sinY_nowaist.cs'

    examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/multicontact-api-master-examples/examples/'
    file_name = 'com_motion_above_feet_WB.cs'

    # Define foot contacts to compute wrench from corner forces
    lxp = 0.1                           # foot length in positive x direction
    lxn = 0.1                           # foot length in negative x direction
    lyp = 0.065                         # foot length in positive y direction
    lyn = 0.065                         # foot length in negative y direction
    # lz = 0.107                          # foot sole height with respect to ankle joint
    lz = 0.0                          # foot sole height with respect to ankle joint

    contact_Point = np.ones((3,4))
    contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
    contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]
    contact_Point[2, :] = 4*[-lz]

    def f12TOphi(f12, points=contact_Point):
        phi = pin.Force.Zero()
        for i in range(4):
            phii = pin.Force(f12[i*3:i*3+3],np.zeros(3))
            fMi = pin.SE3(np.eye(3),points[:,i])
            phi += fMi.act(phii)
        return phi.vector




cs = ContactSequence()
cs.loadFromBinary(examples_dir + file_name)


q_traj   = cs.concatenateQtrajectories()
dq_traj  = cs.concatenateDQtrajectories()
ddq_traj = cs.concatenateDDQtrajectories()
contact_frames = cs.getAllEffectorsInContact()
tau_traj = cs.concatenateTauTrajectories()
print(contact_frames)


min_ts = q_traj.min()
max_ts = q_traj.max() - 0.1
print('traj dur (s): ', max_ts - min_ts)


dt = 1e-3  # discretization timespan
t_arr   = np.arange(min_ts, max_ts, dt)
N = t_arr.shape[0]  # nb of discretized timestamps
q_arr   = np.array([q_traj(t) for t in t_arr])
dq_arr  = np.array([dq_traj(t) for t in t_arr])
ddq_arr = np.array([ddq_traj(t) for t in t_arr])
tau_arr = np.array([tau_traj(t) for t in t_arr])

f_traj_lst = [cs.concatenateContactForceTrajectories(l) for l in contact_frames]
if QUADRUPED:
    f_arr_lst = [np.array([f_traj_lst[i](t) for t in t_arr]) for i in range(len(contact_frames))]
else:
    f_arr_lst = [np.array([f12TOphi(f_traj_lst[i](t)) for t in t_arr]) for i in range(len(contact_frames))]



rmodel = robot.model
rdata = robot.data
contact_frame_ids = [rmodel.getFrameId(l) for l in contact_frames]


if DISPLAY_TRAJ: 
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)

    display_every = 5
    display_dt = display_every * dt
    t1 = time.time()
    for i, q in enumerate(q_arr):
        if (i % display_every) == 0:
            t1 = time.time()
            robot.display(q)
            time_elapsed = time.time() - t1
            if time_elapsed < display_dt: 
                time.sleep(display_dt - time_elapsed)




##################
# FORCE ESTIMATORS
##################

# Joint torques -> force estimation
def estimate_forces(robot, q, vq, dvq, tau_j, cf_ids, dim, world_frame=True):
    """
    dim: 3 -> point contact == 3D force, 6 -> planar contact == 6D wrench
    """
    robot.forwardKinematics(q)

    # simplify the assumption:
    # No linear velocity   --> almost no influence
    # vq[0:3] = 0
    # No angular velocity  --> almost no influence
    # vq[3:6] = 0
    # No joint velocity    --> max 1N difference during flight phase and high variation (lag of the est)
    # vq[6:] = 0
    # No linear spatial acceleration  --> import difference on all axes during the whole traj(~ 10 N diff)
    # dvq[:3] = 0
    # No angular spatial acceleration  --> difference spikes (~ 5N when entering flight phase)
    # dvq[3:6] = 0
    # No joint acceleration            --> ~ 10-20 N diference uniquelly during flight phase
    # dvq[6:] = 0

    tau_cf = pin.rnea(rmodel, rdata, q, vq, dvq)
    tau_cf[6:] -= tau_j
    
    # compute/stack contact jacobians
    Jlinvel = compute_joint_jac(robot, q, cf_ids, dim, world_frame=world_frame)
    
    forces = np.linalg.pinv(Jlinvel.T) @ tau_cf

    return forces.reshape((len(cf_ids),dim)).T


def estimate_torques(robot, q, vq, dvq, forces, cf_ids, dim, world_frame=True):
    robot.forwardKinematics(q)
    tau_tot = pin.rnea(rmodel, rdata, q, vq, dvq)
    
    # compute/stack contact jacobians
    Jlinvel = compute_joint_jac(robot, q, cf_ids, dim, world_frame=world_frame)
    
    tau_forces = Jlinvel.T @ forces 

    return (tau_tot - tau_forces)[6:]


def compute_joint_jac(robot, q, cf_ids, dim, world_frame=True):
    ncf = len(cf_ids)
    Jlinvel = np.zeros((dim*ncf, robot.nv))
    for i, frame_id in enumerate(contact_frame_ids):
        f_Jf = robot.computeFrameJacobian(q, frame_id)[:dim,:]
        if world_frame:
            oXl = robot.framePlacement(q, frame_id, update_kinematics=False).action
            Jlinvel[dim*i:dim*(i+1),:] = oXl[:dim,:dim] @ f_Jf  # jac in world coord
        else: 
            Jlinvel[dim*i:dim*(i+1),:] = f_Jf  # jac in local coord
    return Jlinvel



# store the results of the 3 estimators
f_est_arr_lst = [np.zeros((N,DIMF)) for _ in range(len(contact_frames))]
for i, (q, dq, ddq, tau) in enumerate(zip(q_arr, dq_arr, ddq_arr, tau_arr)):
    # estimate forces
    o_forces = estimate_forces(robot, q, dq, ddq, tau, contact_frame_ids, DIMF, world_frame=WORLD_FRAME)

    for foot_nb in range(len(contact_frames)):
       f_est_arr_lst[foot_nb][i,:] = o_forces[:,foot_nb]

#########################
# plot forces estimations
#########################
NUMBER_OF_LEGS_TO_PLOT = 1

for foot_nb in range(NUMBER_OF_LEGS_TO_PLOT):
    # Forces on each axis
    f, ax = plt.subplots(3,1)
    f.canvas.set_window_title('Force estimation RNEA ' + contact_frames[foot_nb])
    for i, (c, axis) in enumerate(zip('rgb', 'xyz')):
        ax[i].plot(t_arr, f_arr_lst[foot_nb][:,i], c, label='f{} gtr'.format(axis))
        ax[i].plot(t_arr, f_est_arr_lst[foot_nb][:,i], c+'--', label='f{} est'.format(axis))    
    f.suptitle('Force estimation Exact ' + contact_frames[foot_nb])

# Forces error treatment
f_err_arr_lst = [f_est_arr - f_arr for (f_est_arr, f_arr) in zip(f_est_arr_lst, f_arr_lst)]

# Error plot exhibits spikes when each foot touches the ground: the force trajectories seem to be late each time
# Not a bug, cannot be mitigated by damping 
f, ax = plt.subplots(3,1)
for i, (c, axis) in enumerate(zip('rgb', 'xyz')):
    for foot_nb in range(NUMBER_OF_LEGS_TO_PLOT):
        ax[i].plot(t_arr, f_err_arr_lst[foot_nb][:,i],  label='f{} leg{} err'.format(axis, foot_nb))
    plt.legend()
f.suptitle('Force leg est error')


def rmse(arr):
    return np.sqrt(np.mean(arr*2))

# RMSE
print('Root mean squares:')
for foot_nb in range(len(contact_frames)):
    print('l{}: '.format(foot_nb), rmse(f_err_arr_lst[foot_nb]))



#########################
# Joint torque estimation
#########################
tau_joint_est = np.zeros((N,robot.nv-6))
for i, (q, dq, ddq) in enumerate(zip(q_arr, dq_arr, ddq_arr)):
    forces = np.concatenate([f_arr_lst[nbc][i,:] for nbc in range(len(contact_frames))])
    tau_joint_est[i,:] = estimate_torques(robot, q, dq, ddq, forces, contact_frame_ids, DIMF, world_frame=WORLD_FRAME)

plt.figure('Joint torque estimation error')
plt.plot(t_arr, tau_joint_est[:,0] - tau_arr[:,0], 'r--', label='tau0 err')
plt.plot(t_arr, tau_joint_est[:,2] - tau_arr[:,2], 'g--', label='tau2 err')
plt.plot(t_arr, tau_joint_est[:,3] - tau_arr[:,3], 'b--', label='tau3 err')
plt.legend()



plt.show()
