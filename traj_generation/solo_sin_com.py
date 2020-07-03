import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_wrapper import TsidWrapper
import conf_solo12 as conf
from traj_logger import TrajLogger

dt = conf.dt

tsid_solo = TsidWrapper(conf, viewer=True)
logger = TrajLogger(tsid_solo.contact_frame_names, directory='/home/mfourmy/Documents/Phd_LAAS/data/trajs/')

data = tsid_solo.invdyn.data()
robot = tsid_solo.robot

# Params for Com trajectory
pos_f1, _, _ = tsid_solo.get_3d_pos_vel_acc(np.zeros(3), 1)
pos_com_init = robot.com(data)

# Params for Com trajectory
amp        = np.array([-0.02, 0.02, 0.02])                    # amplitude functio
# amp        = np.array([0.0, 0.0, 0.03])                    # amplitude functio
offset     = robot.com(data) - amp                         # offset of the mesured CoM 
two_pi_f             = 2*np.pi*np.array([0.2, 0.4, 0.2])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function

# Init values
q, v = tsid_solo.q, tsid_solo.v
t = 0.0 # time
time_start = time.time()
for i in range(0, conf.N_SIMULATION):
    pos_c = offset + amp * np.cos(two_pi_f*t)
    vel_c = two_pi_f_amp * (-np.sin(two_pi_f*t))
    acc_c = two_pi_f_squared_amp * (-np.cos(two_pi_f*t))
    tsid_solo.set_com_ref(pos_c, vel_c, acc_c)

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
    q, v = tsid_solo.integrate_dv(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_solo.print_solve_check(sol, t, v, dv)

    if (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): 
            time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_solo.update_display(q, t)
        time_start = time.time()



logger.set_data_lst_as_arrays()
# logger.store_csv_trajs('solo_stomping', sep=' ')
# logger.store_mcapi_traj(tsid_solo, 'solo_stomping')

import matplotlib.pyplot as plt

plt.figure('solo stomping contact forces')
plt.title('solo stomping contact forces')
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
plt.show()