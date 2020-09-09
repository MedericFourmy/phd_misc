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
from conf_solo12 import contact_frame_names


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

def forces_from_torques(model, data, q, v, a, tauj, contact_ids):
    Jlinvel = np.zeros((12, 12))
    for i, ct_id in enumerate(contact_ids):
        f_Jf = robot.computeFrameJacobian(q, ct_id)[:3,6:]
        Jlinvel[3*i:3*i+3,:] = f_Jf
    
    tau_tot = pin.rnea(model, data, q, v, a)
    tau_forces = tau_tot[6:] - tauj
    forces = np.linalg.solve(Jlinvel.T, tau_forces)
    return forces

        
DT = 0.001
SLEEP = False

path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'
srdf = path + '/srdf/solo.srdf'
reference_config_q_name = 'standing'

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
cs_file_name = 'solo_sin_traj.cs'
cs_file_path = cs_file_dir + cs_file_name
cs_file_pyb_name = cs_file_name.split('.')[0] + '_pyb'
cs = ContactSequence()
cs.loadFromBinary(cs_file_path)
q_traj = cs.concatenateQtrajectories()
dq_traj = cs.concatenateDQtrajectories()
tau_traj = cs.concatenateTauTrajectories()
ee_names = cs.getAllEffectorsInContact()
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


# q_init = model.referenceConfigurations['standing']
q_init = q_arr[0,:]
dq_init = np.zeros((robot.nv))
controlled_joint_names = []
legs = ['FL', 'FR', 'HL', 'HR']
for leg in legs:
    controlled_joint_names += [leg + '_HAA', leg + '_HFE', leg + '_KFE']

print()
print('ee_names: ', ee_names)
print('contact_frame_names: ', contact_frame_names)

sim = SimulatorPybullet(DT, q_init, 12, robot, controlled_joint_names, contact_frame_names, gravity=[0, 0.0, -9.81])

N = len(q_arr)

print('sim.robotId: ', sim.robotId)
fm = ForceMonitor(sim.robotId, 0)

delays = []
v_prev = np.zeros(18)

# Center the camera on the current position of the robot
pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
                            cameraTargetPosition=[q_init[0], q_init[1], 0.0])


logger = TrajLogger(contact_frame_names, cs_file_dir)
t = 0.0
for i in range(N):
    t1 = time.time()

    # Get position/orientation of the base and angular position of actuators
    sim.retrieve_pyb_data()
    q = sim.qmes.flatten()  # w_p_b, w_q_b, qa
    v = sim.vmes.flatten()  # nu_b, dqa
    dv = (v - v_prev)/DT
    v_prev = v
    c, dc = robot.com(q, v)
    b_h = robot.centroidalMomentum(q, v)
    wRb = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    w_Lc = wRb @ b_h.angular

    # torque to be exactly applied on the robot pyb model joints
    tau = compute_control_torque(q_arr[i,7:], v_arr[i,6:], tau_arr[i,6:], q[7:], v[6:])

    # compute forces that should be exerted on the ee due to these torques
    forces_tau = forces_from_torques(model, data, q, v, dv, tau, contact_frame_ids)
    dic_forces = {'f{}'.format(i): forces_tau[3*i:3*i+3] for i in range(4)}

    data_cs = {
        't': t,
        'q': q,
        'v': v,
        'dv': dv,
        'c': c,
        'dc': dc,
        'tau': tau,
        'Lc': w_Lc
    }
    data_cs.update(dic_forces)
    logger.append_data(data_cs)

    # Center the camera on the current position of the robot
    # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
    #                             cameraTargetPosition=[sim.qmes[0, 0], sim.qmes[1, 0], 0.0])

    # Set control torque for all joints
    pyb.setJointMotorControlArray(sim.robotId, sim.revoluteJointIndices,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # Compute one step of simulation
    pyb.stepSimulation()

    # get contact forces in World frame from pybullet
    # forces_pyb = fm.get_contact_forces(display=True)

    # update pin robot geometry based on simulation
    robot.forwardKinematics(sim.qmes)

    # print()
    # print('forces')
    # for ct_id in forces:
    #     print(ct_id, ' ', forces[ct_id]['wpc'])

    # print('framePlacements')
    # for ct_id in contact_frame_ids:
    #     w_T_c = robot.framePlacement(sim.qmes, ct_id)
    #     print(ct_id, ' ', w_T_c.translation)

    # Wait a bit
    delay = time.time() - t1
    delays.append(delay)
    if SLEEP:
        time.sleep(delay)

    t += DT


logger.store_mcapi_traj(cs_file_pyb_name)

plt.figure()
plt.plot(np.arange(N), np.array(delays))
plt.grid()
plt.show()