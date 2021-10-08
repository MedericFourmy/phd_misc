import sys
import time
import numpy as np
import pinocchio as pin
from tsid_wrapper import TsidWrapper
import conf_solo12 as conf
from traj_logger import TrajLogger


TRAJ_NAME = 'solo_sin_back_down_rots_three'
# TRAJ_NAME = None  # No save
# if 0 -> trunk task deactivated
# conf.w_trunk = 0

dt = conf.dt
tsid_solo = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger(tsid_solo.contact_frame_names, directory='/home/mfourmy/Documents/Phd_LAAS/data/trajs/')

robot = tsid_solo.robot
model = tsid_solo.model
data = tsid_solo.invdyn.data()
q, v = tsid_solo.q, tsid_solo.v

DT = 30.0  # seconds
N_SIMULATION = int(DT/conf.dt)
F = 1/DT  # mvt frequency


# Params for Com trajectory
# amplitude of com mvt
# amp        = np.array([-0.01, 0.03, 0.00])                 
# amp        = np.array([0.02, 0.02, -0.01])  # back, left, down
amp        = np.array([0.01, 0.0, -0.01])  # back, left, down
# amp        = np.array([-0.01, 0.04, 0.03]) 
# amp        = np.array([0.0, 0.0, 0.0]) 
offset     = robot.com(data) - amp                         # offset of the measured CoM 
two_pi_f             = 2*np.pi*F*np.array([2, 2, 2])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function

# Params for trunk orientation trajectory
amp_trunk = np.deg2rad(np.array([15, 10, 15]))   # orientation, numbers in degrees
# amp_trunk = np.deg2rad(np.array([10.0, 10.0, 0.0]))   # orientation, numbers in degrees
two_pi_f_trunk     = 2*np.pi*F*np.array([2, 2, 2])  # movement frequencies along each axis
R_trunk_init = robot.framePosition(data, tsid_solo.trunk_link_id).rotation


# Init values
t = 0.0 # time
time_start = time.time()
# for i in range(0, conf.N_SIMULATION):
for i in range(0, N_SIMULATION):
    # set com ref
    pos_c = offset + amp * np.cos(two_pi_f*t)
    vel_c = two_pi_f_amp * (-np.sin(two_pi_f*t))
    acc_c = two_pi_f_squared_amp * (-np.cos(two_pi_f*t))
    tsid_solo.set_com_ref(pos_c, vel_c, acc_c)

    # set trunk ref
    init_rpy_curr = amp_trunk * np.sin(two_pi_f_trunk*t)
    init_R_curr = pin.rpy.rpyToMatrix(*init_rpy_curr)
    R_trunk = R_trunk_init @ init_R_curr
    tsid_solo.set_trunk_ref(R_trunk)

    # Solve
    sol, HQdata = tsid_solo.compute_and_solve(t, q, v)
    
    # Store values
    if(sol.status!=0):
        HQdata.print_all()
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    dv  = tsid_solo.invdyn.getAccelerations(sol)

    logger.append_data_from_sol(t, q, v, dv, tsid_solo, sol)

    # integrate one step
    q, v = tsid_solo.integrate_dv_R3SO3(q, v, dv, dt)
    # q, v = tsid_solo.integrate_dv(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_solo.print_solve_check(sol, t, v, dv)

    if conf.VIEWER_ON and (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_solo.update_display(q, t)
        time_start = time.time()


if TRAJ_NAME is not None:
    logger.store_csv_trajs(TRAJ_NAME, sep=' ', skip_free_flyer=True)
    # logger.store_mcapi_traj(TRAJ_NAME)

import matplotlib.pyplot as plt

plt.figure('solo sin_com contact forces')
plt.title('solo sin_com contact forces')
plt.subplot(3,1,1)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,0], label='f0x')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,0], label='f1x')
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,0], label='f2x')
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,0], label='f3x')
plt.legend()
plt.grid()
plt.subplot(3,1,2)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,1], label='f0y')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,1], label='f1y')
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,1], label='f2y')
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,1], label='f3y')
plt.legend()
plt.grid()
plt.subplot(3,1,3)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='f0z')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='f1z')
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,2], label='f2z')
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,2], label='f3z')
plt.legend()
plt.grid()
if '--show' in sys.argv:
    plt.show()