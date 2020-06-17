#!/usr/bin/env python
# coding: utf-8

# # Task Space Inverse Dynamics trajectory generation
# Copy of tsid exercizes/notebooks TSID_ex1.ipynb and TSID_ex2.ipynb.  
# Removed tutorial text, tsid plut_utils and commands dependency.  
# Adapted urdf files paths.  
# Added trajectory parsing to multicontact API Contact sequence and file generation.


import matplotlib.pyplot as plt;plt.ion()
import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import os
import time as tmp
import eigenpy
eigenpy.switchToNumpyArray()
import tsid
import pinocchio as pin

import gepetto.corbaserver



GVIEWER_DISPLAY_TRAJ = False
STORE_TRAJ = True
SHOW_PLOTS = False
TRAJ_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
OUTPUT_FILE_NAME = 'sinY_nowaist.cs'
WAIST_TASK = False


#############################
# Definition of the tasks gains and weights and the foot geometry (for contact task)
#############################

lxp = 0.1                           # foot length in positive x direction
lxn = 0.1                           # foot length in negative x direction
lyp = 0.065                         # foot length in positive y direction
lyn = 0.065                         # foot length in negative y direction
lz = 0.107                          # foot sole height with respect to ankle joint
mu = 0.3                            # friction coefficient
fMin = 1.0                          # minimum normal force
fMax = 1000.0                       # maximum normal force

rf_frame_name = "leg_right_6_joint"          # right foot joint name
lf_frame_name = "leg_left_6_joint"           # left foot joint name
contactNormal = np.array([0., 0., 1.])       # direction of the normal to the contact surface

w_com = 1.0                       # weight of center of mass task
w_posture = 0.1                   # weight of joint posture task
w_forceRef = 1e-3                 # weight of force regularization task
w_waist = 1.0                     # weight of waist task

kp_contact = 30.0                 # proportional gain of contact constraint
kp_com = 20.0                     # proportional gain of center of mass task               
kp_waist = 500.0                  # proportional gain of waist task

kp_posture = np.array(                                    # proportional gain of joint posture task
    [ 10. ,  5.  , 5. , 1. ,  1. ,  10.,                  # lleg  #low gain on axis along y and knee
    10. ,  5.  , 5. , 1. ,  1. ,  10.,                    # rleg
    500. , 500.  ,                                        # chest
    50.,   10.  , 10.,  10.,    10. ,  10. , 10. ,  10. , # larm
    50.,   10.  , 10., 10.,    10. ,  10. ,  10. ,  10. , # rarm
    100.,  100.]                                          # head
).T 

dt = 0.001                        # controller time step
PRINT_N = 500                     # print every PRINT_N time steps
RT_VIZ = False                    # if True, call time.sleep update gviewer at the right freq, slower to generate trajectories
DISPLAY_N = 20                    # update robot configuration in viwewer every DISPLAY_N time steps
N_SIMULATION = 10000               # number of time steps simulated
print('traj time length: ', N_SIMULATION*dt)


# Set the path where the urdf file of the robot is registered
path = "/opt/openrobots/share/example-robot-data/robots"
urdf = path + '/talos_data/robots/talos_reduced.urdf'
vector = pin.StdVec_StdString()
vector.extend(item for item in path)
# Create the robot wrapper from the urdf, it will give the model of the robot and its data
robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
srdf = path + '/talos_data/srdf/talos.srdf'

if GVIEWER_DISPLAY_TRAJ:
    # Creation of the robot wrapper for gepetto viewer (graphical interface)
    robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [path, ], pin.JointModelFreeFlyer())
    cl = gepetto.corbaserver.Client()
    gui = cl.gui
    robot_display.initViewer(loadModel=True)


# Take the model of the robot and load its reference configurations
model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)
# Set the current configuration q to the robot configuration half_sitting
q = model.referenceConfigurations['half_sitting']
# Set the current velocity to zero
v = np.zeros(robot.nv)


# Display the robot in Gepetto Viewer in the configuration q = halfSitting
if GVIEWER_DISPLAY_TRAJ:
    robot_display.displayCollisions(False)
    robot_display.displayVisuals(True)
    robot_display.display(q)


# Check that the frames of the feet exist. 
assert model.existFrame(rf_frame_name)
assert model.existFrame(lf_frame_name)


# Creation of the inverse dynamics HQP problem using 
# the robot accelerations (base + joints) and the contact forces as decision variables
# As presented in the cell on QP optimisation
invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
# Compute the problem data with a solver based on EiQuadProg: a modified version of uQuadProg++ working with Eigen
invdyn.computeProblemData(0, q, v)
# Get the data -> initial data
data = invdyn.data()


#############################
# Tasks Definitions 
#############################

# In this exercise, for the equilibrium of the CoM, we need 3 task motions:
# 
#  - **TaskComEquality** as **constraint of the control** (priority level = 0) to maintain the equilibrium of the CoM (by following a reference trajectory). It is the most important task so has a weight of 1 (in constraint scope).
#  - **TaskSE3Equality** in the **cost function** (priority level = 1) for the waist of the robot, to maintain its orientation (with a reference trajectory). It is an important task so has a weight of 1 (in cost function scope).
#  - **TaskJointPosture** in the **cost function** (priority level = 1) for the posture of the robot, to maintain it to half-sitting (with a reference trajectory). It is the less important task so has a weight of 0.1 (in cost function scope).


# COM Task
comTask = tsid.TaskComEquality("task-com", robot)
comTask.setKp(kp_com * np.ones(3)) # Proportional gain defined before = 20
comTask.setKd(2.0 * np.sqrt(kp_com) * np.ones(3)) # Derivative gain = 2 * sqrt(20)
# Add the task to the HQP with weight = 1.0, priority level = 0 (as real constraint) and a transition duration = 0.0
invdyn.addMotionTask(comTask, w_com, 0, 0.0)


# WAIST Task
waistTask = tsid.TaskSE3Equality("keepWaist", robot, 'root_joint') # waist -> root_joint
waistTask.setKp(kp_waist * np.ones(6)) # Proportional gain defined before = 500
waistTask.setKd(2.0 * np.sqrt(kp_waist) * np.ones(6)) # Derivative gain = 2 * sqrt(500), critical damping

# Add a Mask to the task which will select the vector dimensions on which the task will act.
# In this case the waist configuration is a vector 6d (position and orientation -> SE3)
# Here we set a mask = [0 0 0 1 1 1] so the task on the waist will act on the orientation of the robot
mask = np.ones(6)
mask[:3] = 0.
waistTask.setMask(mask)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
if WAIST_TASK:
    invdyn.addMotionTask(waistTask, w_waist, 1, 0.0)


# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture) # Proportional gain defined before (different for each joints)
postureTask.setKd(2.0 * kp_posture) # Derivative gain = 2 * kp
# Add the task to the HQP with weight = 0.1, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)


#############################
## Rigid Contacts Definitions 
#############################

# CONTACTS 6D
# Definition of the foot geometry with respect to the ankle joints (which are the ones controlled)
contact_Point = np.ones((3,4))
contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]
contact_Point[2, :] = 4*[-lz]

# The feet are the only bodies in contact in this experiment and their geometry define the plane of contact
# between the robot and the environement -> it is a Contact6D

# To define a contact6D :
# We need the surface of contact (contact_Point), the normal vector of contact (contactNormal along the z-axis)
# the friction parameter with the ground (mu = 0.3), the normal force bounds (fMin =1.0, fMax=1000.0) 

# Right Foot 
contactRF = tsid.Contact6d("contact_rfoot", robot, rf_frame_name, contact_Point, contactNormal, mu, fMin, fMax)
contactRF.setKp(kp_contact * np.ones(6).T) # Proportional gain defined before = 30
contactRF.setKd(2.0 * np.sqrt(kp_contact) * np.ones(6)) # Derivative gain = 2 * sqrt(30)
# Reference position of the right ankle -> initial position  
H_rf_ref = robot.position(data, model.getJointId(rf_frame_name))
contactRF.setReference(H_rf_ref)
# Add the contact to the HQP with weight = 0.1, priority level = 1e-3 (as real constraint)
invdyn.addRigidContact(contactRF, w_forceRef)

# Left Foot
contactLF = tsid.Contact6d("contact_lfoot", robot, lf_frame_name, contact_Point, contactNormal, mu, fMin, fMax)
contactLF.setKp(kp_contact * np.ones(6).T) # Proportional gain defined before = 30
contactLF.setKd(2.0 * np.sqrt(kp_contact) * np.ones(6).T) # Derivative gain = 2 * sqrt(30)
# Reference position of the left ankle -> initial position  
H_lf_ref = robot.position(data, model.getJointId(lf_frame_name))
contactLF.setReference(H_lf_ref)
# Add the contact to the HQP with weight = 0.1, priority level = 1e-3 (as real constraint)
invdyn.addRigidContact(contactLF, w_forceRef)


# ## TSID Trajectory
# 
# A **Trajectory** is a multi-dimensional function of time describing the motion of an object and its time derivatives.
# For standard use in control, the method *compute_next* is provided, which computes the value of the trajectory at the next time step.
# 
# In the example, we need to set 3 trajectories, one for each task. 
# These trajectories will give at each time step the desired position, velocity and acceleration of the different tasks (CoM, posture and waist).
# In our case they will be constants, equal to their initial values.
# 


# Set the reference trajectory of the tasks
com_ref = data.com[0] # Initial value of the CoM
# com_ref[1] += 0.1
trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref) 
sampleCom = trajCom.computeNext() # Compute the first step of the trajectory from the initial value


qa_ref = q[7:] # Initial value of the joints of the robot (in halfSitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", qa_ref)

waist_ref = robot.position(data, model.getJointId('root_joint')) # Initial value of the waist (root_joint)
# Here the waist is defined as a 6d vector (position + orientation) so it is in the SE3 group (Lie group)
# Thus, the trajectory is not Euclidian but remains in the SE3 domain -> TrajectorySE3Constant
trajWaist = tsid.TrajectorySE3Constant("traj_waist", waist_ref)



# Initialisation of the Solver
# Use EiquadprogFast: dynamic matrix sizes (memory allocation performed only when resizing)
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn) 



# Initialisation of the plot variables which will be updated during the simulation loop 
# These variables describe the behavior of the CoM of the robot (reference and real position, velocity and acceleration)
# reference trajectory and task desired CoM acceleration
com_pos_ref = np.zeros((3, N_SIMULATION))
com_vel_ref = np.zeros((3, N_SIMULATION))
com_acc_ref = np.zeros((3, N_SIMULATION))
com_acc_des = np.zeros((3, N_SIMULATION)) 

# trajectories computed from sequential integration
t_traj   = np.zeros(N_SIMULATION)
q_traj   = np.zeros((robot.nq, N_SIMULATION))
dq_traj  = np.zeros((robot.nv, N_SIMULATION))
ddq_traj = np.zeros((robot.nv, N_SIMULATION))
tau_traj = np.zeros((robot.nv-6, N_SIMULATION))

com_pos = np.zeros((3, N_SIMULATION))
com_vel = np.zeros((3, N_SIMULATION))
com_acc = np.zeros((3, N_SIMULATION))

f_lf_traj = np.zeros((12, N_SIMULATION))
f_rf_traj = np.zeros((12, N_SIMULATION))

Lc_traj = np.zeros((3, N_SIMULATION))



# COM trajectory parameters (if sinusoid)
# PROBLEM: not the right one, transitory moment that is strongly reflected in estimation for some reason
# offset     = robot.com(data)                               # offset of the mesured CoM 
# amp        = np.array([0.0, 0.05, 0.0])                    # amplitude function of 0.05 along the y axis 
# two_pi_f             = 2*np.pi*np.array([0.0, 0.5, 0.0])   # 2π function along the y axis with 0.5 amplitude
# two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
# two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function

amp        = np.array([0.0, 0.03, 0.00])                    # amplitude functio
offset     = robot.com(data) - amp                         # offset of the mesured CoM 
two_pi_f             = 2*np.pi*np.array([0.0, 0.2, 0.0])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function




# Simulation loop
# At each time step compute the next desired trajectory of the tasks
# Set them as new references for each tasks 
# Compute the new problem data (HQP problem update)
# Solve the problem with the solver

# Get the forces and the accelerations computed by the solver
# Update the plot variables of the CoM 
# Print the forces applied at each feet 
# Print the tracking error of the CoM task and the norm of the velocity and acceleration needed to follow the 
# reference trajectory

# Integrate the control (which is in acceleration and is given to the robot in position):
# One simple euler integration from acceleration to velocity
# One integration (velocity to position) with pinocchio to have the freeFlyer updated
# Display the result on the gepetto viewer

t = 0.0 # time
for i in range(0, N_SIMULATION):
    # sampleCom = trajCom.computeNext()
    # comTask.setReference(sampleCom)
        
    sampleCom.pos(offset + amp * np.cos(two_pi_f*t))
    sampleCom.vel(two_pi_f_amp * (-np.sin(two_pi_f*t)))
    sampleCom.acc(two_pi_f_squared_amp * (-np.cos(two_pi_f*t)))
    comTask.setReference(sampleCom)
    
    samplePosture = trajPosture.computeNext()
    postureTask.setReference(samplePosture)
    
    sampleWaist = trajWaist.computeNext()
    waistTask.setReference(sampleWaist)

    HQPData = invdyn.computeProblemData(t, q, v)

    sol = solver.solve(HQPData)
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    tau = invdyn.getActuatorForces(sol)
    dv = invdyn.getAccelerations(sol)
    
    com_pos_ref[:,i] = sampleCom.pos()
    com_vel_ref[:,i] = sampleCom.vel()
    com_acc_ref[:,i] = sampleCom.acc()
    com_acc_des[:,i] = comTask.getDesiredAcceleration
    
    # actual simulated trajectory and controls
    t_traj[i] = t
    q_traj[:,i] = q
    dq_traj[:,i] = v
    ddq_traj[:,i] = dv
    tau_traj[:,i] = tau
    
    com_pos[:,i] = robot.com(invdyn.data())
    com_vel[:,i] = robot.com_vel(invdyn.data())
    com_acc[:,i] = comTask.getAcceleration(dv)
    f_lf_traj[:,i] = invdyn.getContactForce(contactLF.name, sol)
    f_rf_traj[:,i] = invdyn.getContactForce(contactRF.name, sol)
    Lc_traj[:,i] = pin.computeCentroidalMomentum(robot.model(),robot.data(),q,v).angular
    

    if i%PRINT_N == 0:
        print ("Time %.3f"%(t))
        if invdyn.checkContact(contactRF.name, sol):
            f = invdyn.getContactForce(contactRF.name, sol)
            print ("\tnormal force %s: %.1f"%(contactRF.name.ljust(20,'.'),contactRF.getNormalForce(f)))

        if invdyn.checkContact(contactLF.name, sol):
            f = invdyn.getContactForce(contactLF.name, sol)
            print ("\tnormal force %s: %.1f"%(contactLF.name.ljust(20,'.'),contactLF.getNormalForce(f)))
 
        print ("\ttracking err %s: %.3f"%(comTask.name.ljust(20,'.'),       norm(comTask.position_error, 2)))
        print ("\t||v||: %.3f\t ||dv||: %.3f"%(norm(v, 2), norm(dv)))


    if i==N_SIMULATION-1:
        print('End of simulation')
        break

    # Integration scheme used in MCAPI
    # v_mean = v + 0.5*dt*dv
    # v += dt*dv
    # q = pin.integrate(model, q, dt*v_mean)
    ##########################

    # Integration scheme used in WOLF (R3xSO(3) in WOLF though)
    # SE3
    q = pin.integrate(model, q, dt*v)
    v += dt*dv
    t += dt
    #########################

    # # DIRECT INTEGRATION
    # b_v = v[0:3]
    # b_w = v[3:6]
    # b_acc = dv[0:3] + np.cross(b_w, b_v)
    
    # p_int = q[:3]
    # oRb_int = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    # v_int = oRb_int@v[:3]

    # p_int = p_int + v_int*dt + 0.5*oRb_int @ b_acc*dt**2
    # v_int = v_int + oRb_int @ b_acc*dt
    # oRb_int = oRb_int @ pin.exp(b_w*dt)

    # q[:3] = p_int
    # q[3:7] = pin.Quaternion(oRb_int).coeffs()
    # q[7:] += v[6:]*dt
    # v += dt*dv
    # v[:3] = oRb_int.T@v_int

    # t += dt
    # ##########################

if GVIEWER_DISPLAY_TRAJ:
    for i in range(N_SIMULATION):
        time_start = tmp.time()
        q = q_traj[:,i]
        if i%DISPLAY_N == 0: 
            robot_display.display(q)
        
        time_spent = tmp.time() - time_start
        if(time_spent < dt): tmp.sleep(dt-time_spent)
    # Display trajectory in gepetto-gui


# # ##################################
# # ##################################
# # # Verifications on computed wrenches
# # ##################################

# pin.switchToNumpyArray()
# com = robot.com(invdyn.data())
# vcom = robot.com_vel(invdyn.data())
# fl = invdyn.getContactForce(contactLF.name, sol)
# fr = invdyn.getContactForce(contactRF.name, sol)

# tau = invdyn.getActuatorForces(sol)
# a = invdyn.getAccelerations(sol)

# rmodel = robot.model()
# rdata  = rmodel.createData()
# M = pin.crba(rmodel,rdata,q)
# b = pin.nle(rmodel,rdata,q,v)

# ifl = rmodel.getFrameId(lf_frame_name)
# ifr = rmodel.getFrameId(rf_frame_name)

# pin.framesForwardKinematics(rmodel,rdata,q)
# pin.centerOfMass(rmodel,rdata,q)

# def f12TOphi(f12,points=contact_Point):
#     phi = pin.Force.Zero()
#     for i in range(4):
#         phii = pin.Force(f12[i*3:i*3+3],np.zeros(3))
#         fMi = pin.SE3(np.eye(3),points[:,i])
#         phi += fMi.act(phii)
#     return phi

# phil = f12TOphi(fl).vector
# phir = f12TOphi(fr).vector

# Jr = pin.computeFrameJacobian(rmodel,rdata,q,ifr)
# Mr = rdata.oMf[ifr]
# Jl = pin.computeFrameJacobian(rmodel,rdata,q,ifl)
# Ml = rdata.oMf[ifl]

# St = np.vstack([np.zeros([6,rmodel.nv-6]),np.eye(rmodel.nv-6)])

# error = M@a+b - (St@tau+Jr.T@phir+Jl.T@phil)
# assert(norm(error)<1e-8)

# c1=(St@tau+Jr.T@phir+Jl.T@phil)[:6]
# c1=pin.Force(c1[:3],c1[3:])

# wrench_in1 = (M@a+b)[:6]
# wrench_in1 = pin.Force(wrench_in1[:3],wrench_in1[3:])
# cM1 = pin.SE3( rdata.oMi[1].rotation,-rdata.com[0]+rdata.oMi[1].translation)

# weight = rdata.mass[0]*pin.Force(rmodel.gravity.linear,rmodel.gravity.angular)
# cw   =cM1.act(wrench_in1)+weight
# cwbis=pin.computeCentroidalMomentumTimeVariation(rmodel,rdata,q,v,a)

# assert(norm((cw-cwbis).vector)<1e-6)


# ##########################


# ###################################
# ###################################
# # generate wrench trajectory + viz
# ###################################

from example_robot_data import loadTalos
wrench_lf_traj = np.zeros((6,N_SIMULATION))  # left leg wrench
wrench_rf_traj = np.zeros((6,N_SIMULATION))  # right leg wrench

# linear map from redundant collection of 3D forces to 6D wrench
gen_mat_lf = contactLF.getForceGeneratorMatrix
gen_mat_rf = contactRF.getForceGeneratorMatrix

for i in range(N_SIMULATION):
    # gen_mat @ f must compute the wrench at the ankle
    wrench_lf_traj[:,i] = gen_mat_lf @ f_lf_traj[:, i]
    wrench_rf_traj[:,i] = gen_mat_rf @ f_rf_traj[:, i]

# Store the wrench trajectory in a file
import pandas as pd
df_wrench_lf = pd.DataFrame(wrench_lf_traj.T, columns=['f_lfx', 'f_lfy', 'f_lfz', 'tau_lfx', 'tau_lfy', 'tau_lfz'])
df_wrench_rf = pd.DataFrame(wrench_rf_traj.T, columns=['f_rfx', 'f_rfy', 'f_rfz', 'tau_rfx', 'tau_rfy', 'tau_rfz'])
df_wrenches = pd.concat([df_wrench_lf, df_wrench_rf], axis=1)
df_wrenches.to_csv('wre_tsid.csv')





###########################
###########################
# Serialize trajectories in MCAPI file format
###########################

if STORE_TRAJ:
    from multicontact_api import ContactSequence, ContactPhase, ContactPatch
    from curves import piecewise


    cs = ContactSequence()
    cp = ContactPhase()
    cp.timeInitial = t_traj[0]
    cp.timeFinal = t_traj[-1]
    cp.duration = t_traj[-1] - t_traj[0] 

    # assign trajectories :
    t_traj_arr = np.asarray(t_traj).flatten()
    cp.q_t = piecewise.FromPointsList(q_traj, t_traj_arr)
    cp.dq_t = piecewise.FromPointsList(dq_traj, t_traj_arr)
    cp.ddq_t = piecewise.FromPointsList(ddq_traj, t_traj_arr)
    cp.tau_t = piecewise.FromPointsList(tau_traj, t_traj_arr)
    cp.c_t = piecewise.FromPointsList(com_pos, t_traj_arr)
    cp.dc_t = piecewise.FromPointsList(com_vel, t_traj_arr)
    # cp.ddc_t = piecewise.FromPointsList(com_acc, t_traj_arr)  # not needed
    cp.L_t = piecewise.FromPointsList(Lc_traj, t_traj_arr)
    # cp.wrench_t = wrench
    # cp.zmp_t = zmp
    # cp.root_t = root

    # contact force trajectories
    cp.addContact("leg_left_sole_fix_joint", ContactPatch(pin.SE3(),0.5))  # dummy placement and friction coeff
    cp.addContact("leg_right_sole_fix_joint",ContactPatch(pin.SE3(),0.5))
    cp.addContactForceTrajectory("leg_left_sole_fix_joint", piecewise.FromPointsList(f_lf_traj, t_traj_arr))
    cp.addContactForceTrajectory("leg_right_sole_fix_joint", piecewise.FromPointsList(f_rf_traj, t_traj_arr))

    cs.append(cp)  # only one contact phase

    cs.saveAsBinary(TRAJ_FOLDER + OUTPUT_FILE_NAME)
    print('Saved ' + TRAJ_FOLDER + OUTPUT_FILE_NAME)



if SHOW_PLOTS:
    # Position tracking of the CoM along the x,y,z axis
    f, ax = plt.subplots(3,1)
    for i, frame_axis in enumerate('xyz'):
        ax[i].plot(t_traj, com_pos[i,:], label='CoM '+frame_axis)
        ax[i].plot(t_traj, com_pos_ref[i,:], 'r:', label='CoM Ref '+frame_axis)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM [m]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)

    # Velocity tracking of the CoM along the x,y,z axis
    f, ax = plt.subplots(3,1)
    for i, frame_axis in enumerate('xyz'):
        ax[i].plot(t_traj, com_vel[i,:], label='CoM Vel '+frame_axis)
        ax[i].plot(t_traj, com_vel_ref[i,:], 'r:', label='CoM Vel Ref '+frame_axis)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM Vel [m/s]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)

    # Acceleration tracking of the CoM along the x,y,z axis
    f, ax = plt.subplots(3,1)
    for i, frame_axis in enumerate('xyz'):
        ax[i].plot(t_traj, com_acc[i,:], label='CoM Acc '+frame_axis)
        ax[i].plot(t_traj, com_acc_ref[i,:], 'r:', label='CoM Acc Ref '+frame_axis)
        ax[i].plot(t_traj, com_acc_des[i,:], 'g--', label='CoM Acc Des '+frame_axis)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM Acc [m/s^2]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
    # finally show the plots
    plt.show(block=False)


plt.show(block=True)







