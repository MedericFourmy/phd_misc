import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_wrapper import TsidWrapper
import conf_solo12 as conf
from traj_logger import TrajLogger

dt = conf.dt

tsid_solo = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger()

data = tsid_solo.invdyn.data()
robot = tsid_solo.robot

# Params for Com trajectory
H_init_f0 = tsid_solo.robot.framePosition(data, tsid_solo.contact_frame_ids[0])
H_init_f1 = tsid_solo.robot.framePosition(data, tsid_solo.contact_frame_ids[1])
H_init_f2 = tsid_solo.robot.framePosition(data, tsid_solo.contact_frame_ids[2])
H_init_f3 = tsid_solo.robot.framePosition(data, tsid_solo.contact_frame_ids[3])
H_init_lst = [H_init_f0, H_init_f1, H_init_f2, H_init_f3]
pos_init_lst = [H.translation for H in H_init_lst]

raised_foot = 0
feet_nb = [0,1,2,3]

pos_com_init = robot.com(data)
pos_c = pos_com_init.copy()

# param for RF traj during "swing phase"
amp        = np.array([0.05, 0.05, -0.05])                    # amplitude function
# amp        = np.array([0.0, 0.0, 0.0])                    # amplitude function
two_pi_f             = 2*np.pi*np.array([0.2, 0.2, 0.2])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function

def compute_other_feet(fnb, feet_nb):
    return [nb for nb in feet_nb if fnb != nb]

def compute_shift_traj(pos_c, support_feet, shift_duration):
    pos_goal = sum(pos_init_lst[i] for i in support_feet)/3
    return np.linspace(pos_c[:2], pos_goal[:2], int(shift_duration/dt))

SHIFT_DURATION = 3
PARTIAL_SUPPORT_DURATION = 5

# Init values
q, v = tsid_solo.q, tsid_solo.v
t = 0.0 # time
i = 0
time_start = time.time()

# State machine flags 
end_traj = False
new_shift = True
full_support = True
partial_support = False
putting_foot_down = False

raised_foot_nb = 0
while not end_traj:
    
    # update end effector trajectory tasks based on state machine
    if new_shift:
        print('\n\n\n\nnew shift')
        support_feet = compute_other_feet(raised_foot_nb, feet_nb)
        shift_traj = compute_shift_traj(pos_c, support_feet, SHIFT_DURATION)
        new_shift = False
        i_shift = 0
    if full_support:
        pos_c[:2] = shift_traj[i_shift]
        i_shift += 1
        if i_shift >= shift_traj.shape[0]:
            print('\n\n\n\nfull support else')
            tsid_solo.remove_contact(raised_foot_nb)
            full_support = False
            partial_support = True
            t_partial = 0
    if partial_support:
        offset = pos_init_lst[raised_foot_nb] - amp                          
        pos_f = offset + amp * np.cos(two_pi_f*(t_partial))
        vel_f = two_pi_f_amp * (-np.sin(two_pi_f*(t_partial)))
        acc_f = two_pi_f_squared_amp * (-np.cos(two_pi_f*(t_partial)))

        tsid_solo.set_foot_3d_ref(pos_f, vel_f, acc_f, raised_foot_nb)

        t_partial += dt
        if t_partial > PARTIAL_SUPPORT_DURATION:
            print('\n\n\n\npartial support if')
            tsid_solo.add_contact(raised_foot_nb)
            raised_foot_nb += 1

            partial_support = False
            new_shift = True
            full_support = True
            
            if raised_foot_nb > 3:
                end_traj = True



    # dummy values, the com position trajectory is linear (bad)
    # hence the control will lag behind. This is just for testing.
    vel_c = np.zeros(3)
    acc_c = np.zeros(3)    
    tsid_solo.set_com_ref(pos_c, vel_c, acc_c)

    # Solve
    sol, HQdata = tsid_solo.compute_and_solve(t, q, v)

    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break

    # used for integration and storing
    dv  = tsid_solo.invdyn.getAccelerations(sol)

    logger.append_data_from_sol(t, q, v, dv, tsid_solo, sol)

    # integrate one step
    q, v = tsid_solo.integrate_dv(q, v, dv, dt)
    t += dt
    i += 1

    if (i % conf.PRINT_N) == 0:
        tsid_solo.print_solve_check(sol, t, v, dv) 

    if conf.VIEWER_ON and (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_solo.update_display(q, t)
        time_start = time.time()


logger.set_data_lst_as_arrays()
logger.store_qv_trajs('solo_stomping', sep=' ')
logger.store_mcapi_traj(tsid_solo, 'solo_stomping')

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