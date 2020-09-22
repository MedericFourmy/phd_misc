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


def get_Jlinvel(robot, q, contact_ids):
    Jlinvel = np.zeros((12, 12))
    for i, ct_id in enumerate(contact_ids):
        f_Jf = robot.computeFrameJacobian(q, ct_id)[:3,6:]
        Jlinvel[3*i:3*i+3,:] = f_Jf
    return Jlinvel


def forces_from_torques(model, data, q, v, dv, tauj, Jlinvel):
    # print(Jlinvel)
    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot[6:] - tauj
    forces = np.linalg.solve(Jlinvel.T, tau_forces)
    return forces

        
DT = 0.001
SLEEP = False
NOFEET = True
# NOFEET = False

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
# cs_file_dir = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
cs_file_dir = './'
cs_file_name = 'solo_nomove.cs'
# cs_file_name = 'solo_sin_y_notrunk.cs'
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
ee_names = cs.getAllEffectorsInContact()  # not taken into account
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

q_init = np.array([-1.24964491e-06,  6.80742469e-05,  2.26246463e-01, -3.76689866e-05,
                   -9.93946134e-07, -9.64261665e-07,  9.99999999e-01,  1.04524409e-01,
                    8.43752750e-01, -1.68050665e+00, -1.04918212e-01,  8.43499359e-01,
                   -1.68004939e+00,  1.04535100e-01, -8.43750114e-01,  1.68050567e+00,
                   -1.04929422e-01, -8.43503995e-01,  1.68004927e+00,])

# q_init = q_arr[0,:]
# q_init[2] += 0.00178   # Optimal to get the lowest BUMP for point feet of ~0 radius
dq_init = np.zeros((robot.nv))

print()
print('contact_frame_names: ', contact_frame_names)

sim = SimulatorPybullet(URDF_NAME, DT, q_init, 12, robot, controlled_joint_names, contact_frame_names, gravity=[0, 0.0, -9.81])

N = len(q_arr)

print('sim.robotId: ', sim.robotId)
fm = ForceMonitor(sim.robotId, 0, sim.pyb_endeff_ids)

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
t_lst = []
tau_lst = []
o_f_tot_lst = []
for i in range(N):
    print()
    print()
    t1 = time.time()
    print('t: ', t)
    # print('q')
    # print(q)
    # print('v')
    # print(v)

    # STORE
    # Get position/orientation of the base and angular position of actuators
    sim.retrieve_pyb_data()
    q = sim.q  # w_p_b, w_q_b, qa
    v = sim.v  # nu_b, va
    dv = (v - v_prev)/DT
    v_prev = v
    c, dc = robot.com(q, v)
    # print('Robot MASS: ', robot.data.mass[0])
    b_h = robot.centroidalMomentum(q, v)
    wRb = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    w_Lc = wRb @ b_h.angular


    # torque to be exactly applied on the robot pyb model joints
    tau = compute_control_torque(q_arr[i,7:], v_arr[i,6:], tau_arr[i,6:], q[7:], v[6:])

    # Center the camera on the current position of the robot
    # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
    #                             cameraTargetPosition=[sim.q[0, 0], sim.q[1, 0], 0.0])

    # Set control torque for all joints
    pyb.setJointMotorControlArray(sim.robotId, sim.bullet_joint_ids,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # SIMULTATE
    # Compute one step of simulation
    pyb.stepSimulation()

    # get contact forces in World frame from pybullet
    forces_pyb = fm.get_contact_forces(display=True)

    # update pin robot geometry based on simulation
    robot.forwardKinematics(sim.q)

    # data BEFORE simulation step
    data_cs = {
        't': t,
        'q': q,
        'v': v,
        'c': c,
        'dc': dc,
        'tau': tau,
        'Lc': w_Lc
    }

    # TODO: think about shifting the force sensors as well
    sim.retrieve_pyb_data()
    q = sim.q  # w_p_b, w_q_b, qa
    v = sim.v  # nu_b, va
    dv = (v - v_prev)/DT
    v_prev = v
    c, dc = robot.com(q, v)
    b_h = robot.centroidalMomentum(q, v)
    wRb = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    w_Lc = wRb @ b_h.angular

    # data AFTER simulation step
    data_cs['dv'] = dv

    # STORE 
    # compute forces that should be exerted on the ee due to these torques
    Jlinvel = get_Jlinvel(robot, q, contact_frame_ids)
    forces_tau = forces_from_torques(model, data, q, v, dv, tau, Jlinvel)
    
    # TEST
    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot[6:] - tau
    # print('TOTO')
    # print(tau_forces - Jlinvel.T@forces_tau)

    #
    # print('tau control')
    # print(tau)
    # print('delta tau  ', tau - torques_for_test)
    dic_forces = {'f{}'.format(i): forces_tau[3*i:3*i+3] for i in range(4)}

    ###############
    # Force reconstruction
    o_f_tot = np.zeros(3)
    for j, c_id in enumerate(contact_frame_ids):
        l_fl = dic_forces['f{}'.format(j)]
        o_R_l = robot.framePlacement(q, c_id).rotation
        o_fl = o_R_l @ l_fl
        o_f_tot += o_fl
    
    # print('o_f_tot:             ', o_f_tot)
    # mg = robot.data.mass[0] * robot.model.gravity.linear
    # print('m*g:                 ', mg)
    # print('mg+o_f_tot:          ', mg+o_f_tot)
    ################

    ###############
    # Forces from pyb
    ftot_pyb = np.zeros(3)
    for ct_id in forces_pyb:
        ftot_pyb += forces_pyb[ct_id]['f']
    print('ftot_pyb')
    print(ftot_pyb)
    ###############


    ################
    # dynamic equation
    Mq = pin.crba(model, data, q)
    Cqdq = pin.computeCoriolisMatrix(model, data, q, v)
    gq = pin.computeGeneralizedGravity(model, data, q)

    q_static = q.copy()
    q_static[:7] = np.array([0]*6+[1])
    gq_static = pin.computeGeneralizedGravity(model, data, q_static)

    # print('gq free lyer')
    # print(gq[:6])
    # print('o_f_tot - gq')
    # print(o_f_tot - gq[:3])
    # print('gq_static free lyer')
    # print(gq_static[:6])
    print('o_f_tot - gq_static:  ', o_f_tot - gq_static[:3])
    print('ftot_pyb - gq_static: ', ftot_pyb - gq_static[:3])

    Mq_a = Mq[6:,:]
    Cqdq_a = Cqdq[6:,:]
    gq_a = gq[6:]

    diff = Mq_a@dv + Cqdq_a@v + gq_a - Jlinvel.T@forces_tau + tau
    # print('diff')
    # print(diff)
    ################

    data_cs.update(dic_forces)
    logger.append_data(data_cs)

    # print(1/0)

    # print()
    # for ct_id in contact_frame_ids:
    #     w_T_c = robot.framePlacement(sim.q, ct_id)
    #     print('w_t_f', ct_id, ' ', w_T_c.translation)

        # start = w_T_c.translation
        # end = start + np.array([0, 0, 0.1])
        # pyb.addUserDebugLine(start, end, lineColorRGB=[
        #     0.0, 1.0, 0.0], lineWidth=4)


    
    # LOGS
    t_lst.append(t)
    tau_lst.append(tau)
    o_f_tot_lst.append(o_f_tot)



    # Wait a bit
    delay = time.time() - t1
    delays.append(delay)
    if SLEEP:
        time.sleep(delay)

    t += DT
    




logger.store_mcapi_traj(cs_file_pyb_name)

# plt.figure()
# plt.plot(np.arange(N), np.array(delays))
# plt.grid()


plt.figure('tau control')

plt.show()