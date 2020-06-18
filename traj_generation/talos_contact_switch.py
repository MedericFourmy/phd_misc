import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_wrapper import TsidWrapper 
import conf_talos as conf
from traj_logger import TrajLogger

dt = conf.dt

tsid_talos = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger()

data = tsid_talos.invdyn.data()
robot = tsid_talos.robot


# Params for Com trajectory
SHIFT_DURATION = 3
posLF, _, _ = tsid_talos.get_3d_pos_vel_acc(np.zeros(3), 0)
print(posLF)
pos_com_init = robot.com(data)
pos_c = pos_com_init.copy()
shift_traj = np.linspace(pos_c[:2], posLF[:2], SHIFT_DURATION/dt)

# param for RF traj during "swing phase"
posRF, _, _ = tsid_talos.get_3d_pos_vel_acc(np.zeros(3), 1)
amp        = np.array([-0.0, 0.2, -0.2])                    # amplitude function
# amp        = np.array([0.0, 0.0, 0.0])                    # amplitude function
offset     = posRF - amp                          
two_pi_f             = 2*np.pi*np.array([0.2, 0.2, 0.2])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function


# Init values
q, v = tsid_talos.q, tsid_talos.v
t = 0.0 # time
time_start = time.time()
for i in range(0, conf.N_SIMULATION):

    # simple switch
    if i < shift_traj.shape[0]:
        pos_c[:2] = shift_traj[i]
    elif tsid_talos.active_contacts[1]:
        tsid_talos.remove_contact(1)
    else:
        pos_RF = offset + amp * np.cos(two_pi_f*(t-SHIFT_DURATION))
        vel_RF = two_pi_f_amp * (-np.sin(two_pi_f*(t-SHIFT_DURATION)))
        acc_RF = two_pi_f_squared_amp * (-np.cos(two_pi_f*(t-SHIFT_DURATION)))
        tsid_talos.set_foot_3d_ref(pos_RF, vel_RF, acc_RF, 1)
    
    # dummy values, the com position trajectory is linear (bad)
    # hence the control will lag behind. This is just for testing.
    vel_c = np.zeros(3)
    acc_c = np.zeros(3)    
    tsid_talos.set_com_ref(pos_c, vel_c, acc_c)

    # Solve
    sol, HQdata = tsid_talos.compute_and_solve(t, q, v)

    # Store values
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    dv  = tsid_talos.invdyn.getAccelerations(sol)

    logger.append_data_from_sol(t, q, v, dv, tsid_talos, sol)

    # integrate one step
    q, v = tsid_talos.integrate_dv(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_talos.print_solve_check(sol, t, v, dv) 

    if conf.VIEWER_ON and (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_talos.update_display(q, t)
        time_start = time.time()



logger.set_data_lst_as_arrays()
# logger.store_qv_trajs('solo_stomping', sep=' ')
# logger.store_mcapi_traj(tsid_talos, 'solo_stomping')

import matplotlib.pyplot as plt

plt.figure('talos contact switch contact forces')
plt.title('talos contact switch contact forces')
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