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
    Jlinvel = np.zeros((12, 18))
    for i, ct_id in enumerate(contact_ids):
        f_Jf = robot.computeFrameJacobian(q, ct_id)[:3,:]
        Jlinvel[3*i:3*i+3,:] = f_Jf

    return Jlinvel


def forces_from_torques(model, data, q, v, dv, tauj, Jlinvel_red):

    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot[6:] - tauj
    # forces = np.linalg.solve(Jlinvel_red.T, tau_forces)   # equivalent
    forces, _, rank, s = np.linalg.lstsq(Jlinvel_red.T, tau_forces, rcond=None)
    return forces


def forces_from_torques_full(model, data, q, v, dv, tauj, Jlinvel):
        
    tau_tot = pin.rnea(model, data, q, v, dv)
    tau_forces = tau_tot
    tau_forces[6:] -= tauj
    forces, _, rank, s = np.linalg.lstsq(Jlinvel.T, tau_forces, rcond=None)
    return forces

        

        
DT = 0.001
SLEEP = False
NOFEET = True

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

# exact initial configuration matching the steady state of PD control based on q_des
q_init = np.array([-2.67343027e-06,  7.24031162e-05,  2.26213751e-01, -3.77063916e-05,
                   -2.10135742e-06, -1.78050529e-06,  9.99999999e-01,  1.04840508e-01,
                    8.43699081e-01, -1.68053305e+00, -1.05235279e-01,  8.43442591e-01,
                   -1.68007415e+00,  1.04864806e-01, -8.43693656e-01,  1.68053013e+00,
                   -1.05257828e-01, -8.43452273e-01,  1.68007469e+00,])
q_des = np.array([ 0.,     0.,     0.235,  0.,     0.,     0.,     1.,     0.1,    0.8,   -1.6,
                 -0.1,    0.8,   -1.6,    0.1,   -0.8,    1.6,   -0.1,   -0.8,    1.6,  ])

v_init = np.zeros(robot.model.nv)

sim = SimulatorPybullet(URDF_NAME, DT, q_init, 12, robot, controlled_joint_names, contact_frame_names, gravity=[0, 0.0, -9.81])

print('sim.robotId: ', sim.robotId)
fm = ForceMonitor(sim.robotId, 0, sim.pyb_ct_frame_ids)

# Center the camera on the current position of the robot
pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-50, cameraPitch=-39.9,
                            cameraTargetPosition=[q_init[0], q_init[1], 0.0])


t = 0.0

# Init for the control torques computations
sim.retrieve_pyb_data()
q = sim.q  # w_p_b, w_q_b, qa
v = sim.v  # nu_b, va
v_prev = v

lines = {}
for i in range(10000):
    print()
    t1 = time.time()
    print('t: ', t)


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
    # tau = compute_control_torque(q_des[7:], np.zeros(12), np.zeros(12), q_init[7:], np.zeros(12))
    tau = compute_control_torque(q_des[7:], np.zeros(12), np.zeros(12), q[7:], v[6:])

    # Set control torque for all joints
    pyb.setJointMotorControlArray(sim.robotId, sim.bullet_joint_ids,
                                controlMode=pyb.TORQUE_CONTROL, 
                                forces=tau)

    # SIMULATE
    # Compute one step of simulation
    pyb.stepSimulation()

    # get contact forces in World frame from pybullet
    forces_pyb = fm.get_contact_forces(display=True)

    sim.retrieve_pyb_data()
    q = sim.q  # w_p_b, w_q_b, qa
    v = sim.v  # nu_b, va
    dv = (v - v_prev)/DT
    v_prev = v
    c, dc = robot.com(q, v)
    b_h = robot.centroidalMomentum(q, v)
    wRb = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
    w_Lc = wRb @ b_h.angular

    # STORE 
    # compute forces that should be exerted on the ee due to these torques
    Jlinvel = get_Jlinvel(robot, q, contact_frame_ids)
    Jlinvel_red = Jlinvel[:,6:]
    forces_tau = forces_from_torques(model, data, q, v, dv, tau, Jlinvel)
    forces_tau_full = forces_from_torques_full(model, data, q, v, dv, tau, Jlinvel)
    # print('forces_tau - forces_tau_full')
    # print(forces_tau - forces_tau_full)
    dic_forces = {'f{}'.format(i): forces_tau[3*i:3*i+3] for i in range(4)}
    # dic_forces = {'f{}'.format(i): forces_tau_full[3*i:3*i+3] for i in range(4)}
    
    ####################
    # FORCE RECONSTRUCTION TESTS
    ####################

    # tau_tot_pyb = pyb.calculateInverseDynamics(sim.robotId, q[:7], v[:6], dv[:6])
    tau_tot = pin.rnea(model, data, q, v, dv)
    # print('POPO')
    # print(tau_tot_pyb - tau_tot)
    # tau_forces = tau_tot[6:] - tau
    # assert(np.allclose(tau_forces, Jlinvel.T@forces_tau))


    ###############
    # Total force from torques in world frame
    o_f_tot = np.zeros(3)
    for j, c_id in enumerate(contact_frame_ids):
        l_fl = dic_forces['f{}'.format(j)]
        o_R_l = robot.framePlacement(q, c_id).rotation
        o_fl = o_R_l @ l_fl
        o_f_tot += o_fl
    
    # Total force from pybullet api in world frame
    ftot_pyb = np.zeros(3)
    for ct_id in forces_pyb:
        ftot_pyb += forces_pyb[ct_id]['f']

    # print('o_f_tot:             ', o_f_tot)
    # mg = robot.data.mass[0] * robot.model.gravity.linear
    # print('m*g:                 ', mg)
    # print('mg+o_f_tot:          ', mg+o_f_tot)

    # print('ftot_pyb')
    # print(ftot_pyb)
    ################

    ################
    # dynamic equation
    Mq = pin.crba(model, data, q)
    Cqdq = pin.computeCoriolisMatrix(model, data, q, v)
    gq = pin.computeGeneralizedGravity(model, data, q)

    q_static = q.copy()
    q_static[:7] = np.array([0]*6+[1])
    gq_static = pin.computeGeneralizedGravity(model, data, q_static)

    # Compare gravity in world frame with total forces
    print('ftot_pyb - gq_static: ', ftot_pyb - gq_static[:3])  # OK
    print('o_f_tot - gq_static:  ', o_f_tot - gq_static[:3])   # NOTOK

    Mq_a = Mq[6:,:]
    Cqdq_a = Cqdq[6:,:]
    gq_a = gq[6:]

    # another test on the dynamics equation (without free flyer 6 first rows)
    assert(np.allclose(Mq_a@dv + Cqdq_a@v + gq_a, Jlinvel_red.T@forces_tau + tau))
    # In fact after, a very short transitory motion, the acceleration and coriolis terms are negligeable as expected
    # if t > .05:
    #     assert(np.allclose(gq_a, Jlinvel.T@forces_tau + tau))
    ################

    #####################
    # Draw debug lines starting from the ee frame, going upward along z
    # print()
    # for ct_id in contact_frame_ids:
    #     w_T_c = robot.framePlacement(sim.q, ct_id)
    #     print('w_t_f', ct_id, ' ', w_T_c.translation)

        # start = w_T_c.translation
        # end = start + np.array([0, 0, 0.1])
        # pyb.addUserDebugLine(start, end, lineColorRGB=[
        #     0.0, 1.0, 0.0], lineWidth=4)

    #####################

    t += DT
