import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_wrapper import TsidWrapper
import conf_talos as conf
from traj_logger import TrajLogger


tsid_talos = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger(tsid_talos.contact_frame_names, '/home/mfourmy/Documents/Phd_LAAS/data/trajs/')

data = tsid_talos.invdyn.data()
robot = tsid_talos.robot

# Params for Com trajectory
amp        = np.array([0.0, 0.03, 0.00])                    # amplitude function
# amp        = np.array([0.0, 0.0, 0.0])                    # amplitude function
offset     = robot.com(data) - amp                         # offset of the mesured CoM 
two_pi_f             = 2*np.pi*np.array([0.0, 0.2, 0.0])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function

# Init values
q, v = tsid_talos.q, tsid_talos.v
dt = conf.dt
t = 0.0 # time
time_start = time.time()

for i in range(0, conf.N_SIMULATION):
    pos_c = offset + amp * np.cos(two_pi_f*t)
    vel_c = two_pi_f_amp * (-np.sin(two_pi_f*t))
    acc_c = two_pi_f_squared_amp * (-np.cos(two_pi_f*t))
    tsid_talos.set_com_ref(pos_c, vel_c, acc_c)

    # Solve
    sol, HQdata = tsid_talos.compute_and_solve(t, q, v)

    # Store values
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    dv  = tsid_talos.invdyn.getAccelerations(sol)

    # record data before integration
    logger.append_data_from_sol(t, q, v, dv, tsid_talos, sol)

    # integrate one step
    q, v = tsid_talos.integrate_dv_R3SO3(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_talos.print_solve_check(sol, t, v, dv) 

    if conf.VIEWER_ON and (i % conf.DISPLAY_N) == 0:
        # HQdata.print_all()
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_talos.update_display(q, t)
        time_start = time.time()

logger.set_data_lst_as_arrays()
# logger.store_csv_trajs('talos_sin_traj', sep=' ')
logger.store_mcapi_traj(tsid_talos, 'talos_sin_traj_R3SO3')

import matplotlib.pyplot as plt

plt.figure('talos sin traj contact forces')
plt.title('talos sin traj contact forces')
plt.subplot(3,1,1)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,0], label='f0x')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,0], label='f1x')
plt.legend()
plt.grid()
plt.subplot(3,1,2)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,1], label='f0y')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,1], label='f1y')
plt.legend()
plt.grid()
plt.subplot(3,1,3)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='f0z')
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='f1z')
plt.legend()
plt.grid()
plt.show()