import time
import numpy as np
import pandas as pd
import pinocchio as pin
import matplotlib.pyplot as plt
import pybullet as pyb  # Pybullet server
from multicontact_api import ContactSequence

from simulator_pybullet import SimulatorPybullet
from force_monitor_solo12_pyb import ForceMonitor
from traj_logger import TrajLogger


def compute_control_torque(qa_d, va_d, tau_d, qa_mes, va_mes):
    Kp = 7
    Kd = 0.5
    # Kd = 2*np.sqrt(Kp)  # NOPE
    err_qa = qa_d - qa_mes
    err_va = va_d - va_mes

    # tau = Kp*err_qa                      # elastic
    tau = Kp*err_qa + Kd*err_va            # good
    # tau = tau_d + Kp*err_qa + Kd*err_va  # torque makes the feet slide
    return tau


def get_Jlinvel(robot, q, contac_ids):
    Jlinvel = np.zeros((12, 18))
    for i, c_id in enumerate(contac_ids):
        f_Jf = robot.computeFrameJacobian(q, c_id)[:3,:]
        Jlinvel[3*i:3*i+3,:] = f_Jf

    return Jlinvel


def forces_from_torques(model, data, q, v, dv, tauj, Jlinvel_red):
    """
    Works only well if the contact kinematics are perfect
    """
    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot[6:] - tauj
    # forces = np.linalg.solve(Jlinvel_red.T, tau_forces)   # equivalent
    forces, _, rank, s = np.linalg.lstsq(Jlinvel_red.T, tau_forces, rcond=None)
    return forces


def forces_from_torques_full(model, data, q, v, dv, tauj, Jlinvel):
    """
    Compromise between the euler equations and contact leg dynamics
    """
    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot
    tau_forces[6:] -= tauj
    forces, _, rank, s = np.linalg.lstsq(Jlinvel.T, tau_forces, rcond=None)
    return forces

        
        
DT = 0.001
SLEEP = False
GUION = False
NOFEET = True
SAVE_CS = True

if NOFEET:
    URDF_NAME = 'solo12_nofeet.urdf'
else:
    URDF_NAME = 'solo12.urdf'
path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/' + URDF_NAME
srdf = path + '/srdf/solo.srdf'
reference_config_q_name = 'standing'

LEGS = ['FL', 'FR', 'HL', 'HR']
if NOFEET:
    contact_frame_names = [leg+'_FOOT_TIP' for leg in LEGS]  # ......... -> solo12_nofeet
else:
    contact_frame_names = [leg+'_ANKLE' for leg in LEGS]  # # same thing as XX_FOOT but contained in pybullet -> solo12

controlled_joint_names = []
for leg in LEGS:
    controlled_joint_names += [leg + '_HAA', leg + '_HFE', leg + '_KFE']

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
model = robot.model
data = robot.data
pin.loadReferenceConfigurations(model, srdf, False)

contact_frame_ids = [model.getFrameId(ct_frame) for ct_frame in contact_frame_names]
# file_qtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_q.dat'
# file_vtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_v.dat'
# file_tautraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_tau.dat'
# df_q = pd.read_csv(file_qtraj, sep=' ', header=None, index_col=0)
# df_v = pd.read_csv(file_vtraj, sep=' ', header=None, index_col=0)
# df_tau = pd.read_csv(file_tautraj, sep=' ', header=None, index_col=0)


# From q traj files
# file_qtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_q.dat'
# file_vtraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_v.dat'
# file_tautraj = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/traj_generation/temp_traj/solo_stamping_tau.dat'
# df_q = pd.read_csv(file_qtraj, sep=' ', header=None, index_col=0)
# df_v = pd.read_csv(file_vtraj, sep=' ', header=None, index_col=0)
# df_tau = pd.read_csv(file_tautraj, sep=' ', header=None, index_col=0)
# q_arr = df_q.to_numpy()
# v_arr = df_v.to_numpy()
# tau_arr = df_tau.to_numpy()

# From ContactSequence file
cs_file_dir = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
# cs_file_name = 'solo_nomove.cs'
cs_file_name = 'solo_sin_y_notrunk.cs'
# cs_file_name = 'solo_sin_y_trunk.cs'
# cs_file_name = 'solo_sin_traj.cs'
# cs_file_name = 'solo_sin_rp+y.cs'
# cs_file_name = 'solo_stamping_nofeet.cs'
cs_file_path = cs_file_dir + cs_file_name
cs_file_pyb_name = cs_file_name.split('.')[0] + '_Pyb'

if NOFEET:
    cs_file_pyb_name += '_Nofeet'

cs = ContactSequence()
cs.loadFromBinary(cs_file_path)
q_traj = cs.concatenateQtrajectories()
dq_traj = cs.concatenateDQtrajectories()
tau_traj = cs.concatenateTauTrajectories()
# ee_names = cs.getAllEffectorsInContact()  # not taken into account
q_arr = []
v_arr = []
tau_arr = []
t = q_traj.min()
while t < q_traj.max():
    q_arr.append(q_traj(t)) 
    v_arr.append(dq_traj(t)) 
    tau_arr.append(tau_traj(t)) 
    t += DT
q_arr = np.array(q_arr)
v_arr = np.array(v_arr)
tau_arr = np.array(tau_arr)

# q_init = np.array([-3.80707953e-06,  8.20657017e-05,  2.26055010e-01, -3.87928948e-05,
#                    -4.05816557e-06,  1.59424082e-06,  9.99999999e-01,  1.05393051e-01,
#                     8.43588548e-01, -1.68060272e+00, -1.05786780e-01,  8.43342731e-01,
#                    -1.68014066e+00,  1.05439013e-01, -8.43589226e-01,  1.68060008e+00,
#                    -1.05833217e-01, -8.43350394e-01,  1.68013856e+00,])

q_init = q_arr[0,:]
q_init[2] += 0.00132   # Optimal to get the lowest BUMP for point feet of ~0 radius

print()
print('contact_frame_names: ', contact_frame_names)

sim = SimulatorPybullet(URDF_NAME, DT, q_init, 12, robot, controlled_joint_names, contact_frame_names, guion=GUION, gravity=[0, 0.0, -9.81])

N = len(q_arr)-1

print('sim.robotId: ', sim.robotId)
fm = ForceMonitor(sim.robotId, 0, sim.pyb_ct_frame_ids)

delays = []
v_prev = np.zeros(18)

# Center the camera on the current position of the robot
pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
                            cameraTargetPosition=[q_init[0], q_init[1], 0.0])


logger = TrajLogger(contact_frame_names, cs_file_dir)
t = 0.0

# Init for the control torques computations
sim.retrieve_pyb_data()
q = sim.q  # w_p_b, w_q_b, qa
v = sim.v  # nu_b, va




lines = {}
forces_dic_lst = {'f{}'.format(i): [] for i in range(4)}
ftot_lst = []
ftot_pyb_lst = []
for i in range(N):
    t1 = time.time()

    # STORE
    # Get position/orientation of the base and angular position of actuators
    sim.retrieve_pyb_data()
    q = sim.q  # w_p_b, w_q_b, qa
    v = sim.v  # nu_b, va
    c, dc = robot.com(q, v)
    # print('Robot MASS: ', robot.data.mass[0])
    b_h = robot.centroidalMomentum(q, v)
    wRb = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    w_Lc = wRb @ b_h.angular

    # torque to be exactly applied on the robot pyb model joints
    tau = compute_control_torque(q_arr[i,7:], v_arr[i,6:], tau_arr[i,6:], q[7:], v[6:])

    # data BEFORE simulation step
    data_cs = {
        't': t,
        'q': q.copy(),
        'v': v.copy(),
        'c': c.copy(),
        'dc': dc.copy(),
        'Lc': w_Lc.copy()
    }

    # Center the camera on the current position of the robot
    # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
    #                             cameraTargetPosition=[sim.q[0, 0], sim.q[1, 0], 0.0])

    # Set control torque for all joints
    pyb.setJointMotorControlArray(sim.robotId, sim.bullet_joint_ids,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # SIMULATE
    # Compute one step of simulation
    pyb.stepSimulation()

    sim.retrieve_pyb_data()
    q2 = sim.q  # w_p_b, w_q_b, qa
    v2 = sim.v  # nu_b, va

    # COMPUTING ACCELERATION
    # dv = (v - v_prev)/DT       # Backward difference
    # dv = (v2 - v)/DT           # Forward difference
    dv = (v2 - v_prev)/(2*DT)  # Central difference
    data_cs['dv'] = dv

    # Updating previous velocity
    v_prev = v

    tau_next = compute_control_torque(q_arr[i,7:], v_arr[i,6:], tau_arr[i+1,6:], q[7:], v[6:])



    # compute forces that should be exerted on the ee due to these torques
    Jlinvel = get_Jlinvel(robot, q, contact_frame_ids)
    Jlinvel_red = Jlinvel[:,6:]
    # forces_tau = forces_from_torques(model, data, q, v, dv, tau, Jlinvel_red)
    # forces_tau = forces_from_torques_full(model, data, q, v, dv, tau, Jlinvel)
    forces_tau = forces_from_torques_full(model, data, q, v, dv, tau_next, Jlinvel)
    dic_forces_tau = {'f{}'.format(i): forces_tau[3*i:3*i+3] for i in range(4)}
    for i in range(4):
        forces_dic_lst['f{}'.format(i)].append(dic_forces_tau['f{}'.format(i)])
    
    ###################################
    # Compute total forces for tests
    # get contact forces in World frame from pybullet
    forces_pyb = fm.get_contact_forces(display=True)
    ftot_tau = np.zeros(3)
    ftot_pyb = np.zeros(3)
    dic_forces_pyb = {}
    # print()
    for j, (c_id_pin, c_id_pyb) in enumerate(zip(contact_frame_ids, fm.contact_links_ids)):
        o_fpyb_l = forces_pyb[c_id_pyb]['f']
        l_ftau_l = dic_forces_tau['f{}'.format(j)]
        
        o_R_l = robot.framePlacement(q, c_id_pin).rotation
        o_ftau_l = o_R_l @ l_ftau_l
        l_fpyb_l = o_R_l.T @ o_fpyb_l

        dic_forces_pyb['f{}'.format(j)] = l_fpyb_l

        ftot_pyb += o_fpyb_l
        ftot_tau += o_ftau_l
    
    ftot_lst.append(ftot_tau)
    ftot_pyb_lst.append(ftot_pyb)
    ###################################

    # storing torque
    # data_cs['tau'] = tau_next  # apparently slightly better, to verify
    data_cs['tau'] = tau
    data_cs.update(dic_forces_tau)
    # data_cs.update(dic_forces_pyb)

    logger.append_data(data_cs)

    #####################
    # Draw debug lines starting from the ee frame, going upward along z
    # print()
    # for c_id in contact_frame_ids:
    #     w_T_c = robot.framePlacement(sim.q, c_id)
    #     print('w_t_f', c_id, ' ', w_T_c.translation)

        # start = w_T_c.translation
        # end = start + np.array([0, 0, 0.1])
        # pyb.addUserDebugLine(start, end, lineColorRGB=[
        #     0.0, 1.0, 0.0], lineWidth=4)

    #####################

    # Wait a bit
    delay = time.time() - t1
    delays.append(delay)
    if SLEEP:
        time.sleep(delay)

    t += DT



if SAVE_CS:
    logger.store_mcapi_traj(cs_file_pyb_name)

# plt.figure()
# plt.plot(np.arange(N), np.array(delays))
# plt.grid()
# plt.show()

t_arr = np.arange(N)*DT

# for i in range(4):
#     plt.figure('f{}'.format(i))
#     fi_arr = np.array(forces_dic_lst['f{}'.format(i)])
#     # plt.plot(t_arr, fi_arr[:,0], 'r.')
#     plt.plot(t_arr, fi_arr[:,1], 'g.')
#     # plt.plot(t_arr, fi_arr[:,2], 'b.')

ftot_arr = np.array(ftot_lst)
ftot_pyb_arr = np.array(ftot_pyb_lst)

f_diff = ftot_arr - ftot_pyb_arr

plt.figure('ftotX')
plt.plot(t_arr, ftot_arr[:,0], 'r.')
plt.plot(t_arr, ftot_arr[:,1], 'g.')
plt.plot(t_arr, ftot_arr[:,2], 'b.')

plt.figure('ftot_pybX')
plt.plot(t_arr, ftot_pyb_arr[:,0], 'r.')
plt.plot(t_arr, ftot_pyb_arr[:,1], 'g.')
plt.plot(t_arr, ftot_pyb_arr[:,2], 'b.')

plt.figure('fdiff')
plt.plot(t_arr, f_diff[:,0], 'r.')
plt.plot(t_arr, f_diff[:,1], 'g.')
plt.plot(t_arr, f_diff[:,2], 'b.')


plt.show()