import sys
import time
import numpy as np
import pinocchio as pin
from tsid_wrapper import TsidWrapper
from traj_logger import TrajLogger
from scipy.stats import logistic

from utils import *

import conf_solo12 as conf

TRAJ_DIR = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
TRAJ_NAME = 'solo_stamping'
SAVE = '--save' in sys.argv
SLEEP = '--sleep' in sys.argv
SHOW = '--show' in sys.argv

dt = conf.dt



tsid_solo = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger(tsid_solo.contact_frame_names, directory=TRAJ_DIR)

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


# TRAJECTORY PARAMETERS
SHIFT_DURATION = 3.0  # time to shift the COM from one support triangle barycentre to the next
PARTIAL_SUPPORT_DURATION = 5.0  # time during which each foot is raised

# amplitudes of the sinusoids followed by 
amp_lst = [
    np.array([0.0,  0.0, -0.05]),
    np.array([0.0, -0.0, -0.05]),
    np.array([0.0,  0.0, -0.05]),
    np.array([0.0, -0.0, -0.05]),
    ]

# a full sinusoid on each axis 
freq_partial = np.array([
    1/PARTIAL_SUPPORT_DURATION,
    1/PARTIAL_SUPPORT_DURATION,
    1/PARTIAL_SUPPORT_DURATION,
])

freq_shift = np.array([
    1/(2*SHIFT_DURATION),
    1/(2*SHIFT_DURATION),
    1/(2*SHIFT_DURATION),
])

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
back_to_origin = False

prev_raised_foot_idx = 0  # foot previously raised
raised_foot_idx = 0
w_prev = conf.w_forceRef
w_next = conf.w_forceRef
x_prev = 0
x_next = 0
w_prev_arr = []
w_next_arr = []
x_prev_arr = []
x_next_arr = []
while not end_traj:

    # update end effector trajectory tasks based on state machine
    if new_shift:
        data = tsid_solo.invdyn.data()
        pos_c = robot.com(data)
        if back_to_origin:
            pos_c_goal = pos_com_init
        else:    
            support_feet = compute_other_feet(raised_foot_idx, feet_nb)
            pos_c_goal = sum(pos_init_lst[i] for i in support_feet)/3
            pos_c_goal[2] = pos_c[2]  # keep z constant?
        
        amp_com = -(pos_c_goal - pos_c)/2
        offset = pos_c - amp_com

        new_shift = False
        t_shift = 0
        dist_shift = dist(pos_c, pos_c_goal)
        ramp_perc = 0.5
        # ramp_perc = 0.0
        dist_max_w_next_ramp = ramp_perc*dist_shift 
        dist_min_w_prev_ramp = (0.6)*dist_shift

    if full_support:
        dist_to_goal = dist(pos_c[:2], pos_c_goal[:2])
        pos_c, vel_c, acc_c = compute_cos_traj(t_shift, amp_com, offset, freq_shift)

        # foot force regularization when switching contacts
        if t_shift < SHIFT_DURATION:

            # start with big regularization when foot just lands
            first_time = prev_raised_foot_idx == raised_foot_idx
            if not first_time and t_shift < 0.5:
                w_prev = linear_interp(t_shift, 0, 0.5, 10*conf.w_forceRef, conf.w_forceRef)
                tsid_solo.contacts[prev_raised_foot_idx].setRegularizationTaskWeightVector(w_prev*np.ones(3))

            # when foot about to rise, bigger regularization
            if not back_to_origin and (dist_to_goal <= dist_max_w_next_ramp):
                w_next = linear_interp(t_shift, SHIFT_DURATION-0.5, SHIFT_DURATION, conf.w_forceRef, 1000*conf.w_forceRef)
                tsid_solo.contacts[raised_foot_idx].setRegularizationTaskWeightVector(w_next*np.ones(3))

            t_shift += dt
        
        else:
            full_support = False
            partial_support = True
            t_partial = 0
            
            if back_to_origin:
                end_traj = True
            else:
                tsid_solo.remove_contact(raised_foot_idx)


    if partial_support and not end_traj:
        offset = pos_init_lst[raised_foot_idx] - amp_lst[raised_foot_idx]                          
        pos_f, vel_f, acc_f = compute_cos_traj(t_partial, amp_lst[raised_foot_idx], offset, freq_partial)

        tsid_solo.set_foot_3d_ref(pos_f, vel_f, acc_f, raised_foot_idx)

        t_partial += dt
        if t_partial > PARTIAL_SUPPORT_DURATION:
            tsid_solo.add_contact(raised_foot_idx)
            prev_raised_foot_idx = raised_foot_idx
            raised_foot_idx += 1

            partial_support = False
            new_shift = True
            full_support = True
            
            if raised_foot_idx > 3:
                back_to_origin = True

    w_prev_arr.append(w_prev)
    w_next_arr.append(w_next)
    x_prev_arr.append(x_prev)
    x_next_arr.append(x_next)

    # dummy values, the com position trajectory is linear (bad)
    # hence the control will lag behind. This is just for testing.
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
    q, v = tsid_solo.integrate_dv_R3SO3(q, v, dv, dt)
    t += dt
    i += 1

    if (i % conf.PRINT_N) == 0:
        tsid_solo.print_solve_check(sol, t, v, dv) 

    if SLEEP and conf.VIEWER_ON and (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_solo.update_display(q, t)
        time_start = time.time()



logger.set_data_lst_as_arrays()
if SAVE:
    logger.store_csv_trajs(TRAJ_NAME, sep=' ', skip_free_flyer=False)

import matplotlib.pyplot as plt


# FORCE
plt.figure('solo stamping contact forces')
plt.title('solo stamping contact forces')
plt.subplot(4,1,1)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,0], label='{}_fx'.format(logger.contact_names[0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,0], label='{}_fx'.format(logger.contact_names[1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,0], label='{}_fx'.format(logger.contact_names[2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,0], label='{}_fx'.format(logger.contact_names[3]))
plt.legend()
plt.grid()
plt.subplot(4,1,2)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,1], label='{}_fy'.format(logger.contact_names[0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,1], label='{}_fy'.format(logger.contact_names[1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,1], label='{}_fy'.format(logger.contact_names[2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,1], label='{}_fy'.format(logger.contact_names[3]))
plt.legend()
plt.grid()
plt.subplot(4,1,3)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='{}_fz'.format(logger.contact_names[0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='{}_fz'.format(logger.contact_names[1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,2], label='{}_fz'.format(logger.contact_names[2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,2], label='{}_fz'.format(logger.contact_names[3]))
plt.legend()
plt.grid()

# TORQUE
plt.figure('solo stamping torques')
plt.title('solo stamping torques')
plt.subplot(4,1,1)
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*0+0], label='{}_tau'.format(3*0+0))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*0+1], label='{}_tau'.format(3*0+1))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*0+2], label='{}_tau'.format(3*0+2))
plt.legend()
plt.grid()
plt.subplot(4,1,2)
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*1+0], label='{}_tau'.format(3*1+0))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*1+1], label='{}_tau'.format(3*1+1))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*1+2], label='{}_tau'.format(3*1+2))
plt.legend()
plt.grid()
plt.subplot(4,1,3)
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*2+0], label='{}_tau'.format(3*2+0))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*2+1], label='{}_tau'.format(3*2+1))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*2+2], label='{}_tau'.format(3*2+2))
plt.legend()
plt.subplot(4,1,4)
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*3+0], label='{}_tau'.format(3*3+0))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*3+1], label='{}_tau'.format(3*3+1))
plt.plot(logger.data_log['t'], logger.data_log['tau'][:,3*3+2], label='{}_tau'.format(3*3+2))
plt.legend()
plt.grid()


o_p_ob = logger.data_log['q'][:,:3]
o_q_b = logger.data_log['q'][:,3:7]
qa = logger.data_log['q'][:,7:]

plt.figure('Robot position')
plt.title('Robot position')
for k in range(3):
    plt.subplot(4,1,1)
    plt.plot(logger.data_log['t'], o_p_ob[:,k], 'rgb'[k])

plt.figure('Feet contacts and normal forces')
plt.subplot(2,1,1)
plt.title('Contacts')
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,0], label='{}_contact'.format(logger.contact_names[0]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,1], label='{}_contact'.format(logger.contact_names[1]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,2], label='{}_contact'.format(logger.contact_names[2]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,3], label='{}_contact'.format(logger.contact_names[3]))
plt.subplot(2,1,2)
plt.title('Normal forces (N)')
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='{}_fz'.format(logger.contact_names[0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='{}_fz'.format(logger.contact_names[1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,2], label='{}_fz'.format(logger.contact_names[2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,2], label='{}_fz'.format(logger.contact_names[3]))
plt.legend()


plt.figure()
plt.plot(logger.data_log['t'], x_prev_arr, label='x_prev')
plt.plot(logger.data_log['t'], x_next_arr, label='x_next')
plt.legend()


if SHOW:
    plt.show()