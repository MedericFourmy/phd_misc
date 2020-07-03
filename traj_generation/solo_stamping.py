import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_wrapper import TsidWrapper
import conf_solo12 as conf
from traj_logger import TrajLogger
from scipy.stats import logistic

dt = conf.dt

tsid_solo = TsidWrapper(conf, viewer=conf.VIEWER_ON)
logger = TrajLogger(tsid_solo.contact_frame_names, directory='temp_traj')

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


def compute_other_feet(fnb, feet_nb):
    return [nb for nb in feet_nb if fnb != nb]

def compute_traj_shift(pos_c, support_feet, shift_duration):
    pos_goal = sum(pos_init_lst[i] for i in support_feet)/3
    return np.linspace(pos_c[:2], pos_goal[:2], int(shift_duration/dt))

def linear_interp(x, xa, xb, ya, yb):
    return ya + (x-xa)*(yb-ya)/(xb-xa)

def log_linear_interp(x, xa, xb, ya, yb):
    pass

def dist(posa, posb):
    return np.linalg.norm(posa-posb)

def compute_cos_traj(t, amp, freq):
    # param for RF traj during "swing phase"
    two_pi_f             = 2*np.pi*freq   # movement frequencies along each axis
    two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
    two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function
    offset = pos_init_lst[raised_foot_nb] - amp                          
    pos_f = offset + amp * np.cos(two_pi_f*(t_partial))
    vel_f = two_pi_f_amp * (-np.sin(two_pi_f*(t_partial)))
    acc_f = two_pi_f_squared_amp * (-np.cos(two_pi_f*(t_partial)))
    return pos_f, vel_f, acc_f 


SHIFT_DURATION = 3.0
PARTIAL_SUPPORT_DURATION = 5.0

amp_lst = [
    np.array([0.02,  0.05, -0.05]),
    np.array([0.02, -0.05, -0.05]),
    np.array([0.02,  0.05, -0.05]),
    np.array([0.02, -0.05, -0.05]),
    ]
# req = np.array([0.2, 0.2, 0.2])
# freq = np.ones(3)/PARTIAL_SUPPORT_DURATION
freq = np.array([
    1/PARTIAL_SUPPORT_DURATION,
    1/PARTIAL_SUPPORT_DURATION,
    1/PARTIAL_SUPPORT_DURATION,
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

prev_raised_foot_nb = 0
raised_foot_nb = 0
w_forceRef_big = 70*conf.w_forceRef
w_forceRef_big_newcontact = 70*conf.w_forceRef
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
        print('\n\n\n\nnew shift')
        support_feet = compute_other_feet(raised_foot_nb, feet_nb)
        traj_shift = compute_traj_shift(pos_c, support_feet, SHIFT_DURATION)
        new_shift = False
        i_shift = 0
        dist_shift = dist(traj_shift[0,:], traj_shift[-1,:])
        ramp_perc = 0.5
        dist_max_w_next_ramp = ramp_perc*dist_shift 
        dist_min_w_prev_ramp = (0.8)*dist_shift


    if full_support:
        pos_c[:2] = traj_shift[i_shift]
        dist_to_goal = dist(pos_c[:2], traj_shift[-1,:])

        if i_shift < traj_shift.shape[0]-1:
            # handle force regularization tasks weights for smooth transitions
            if (prev_raised_foot_nb != raised_foot_nb) and (dist_to_goal >= dist_min_w_prev_ramp):
                #
                # w_prev = logistic.cdf(dist_max_w_next_ramp-dist_to_goal, dist_max_w_next_ramp/2, dist_max_w_next_ramp/20)
                #
                # w_prev = linear_interp(dist_to_goal, dist_shift, dist_min_w_prev_ramp, w_forceRef_big, conf.w_forceRef)
                #
                # print(dist_to_goal-dist_min_w_prev_ramp, ' / ', dist_min_w_prev_ramp)
                x_prev = logistic.cdf(dist_to_goal-dist_min_w_prev_ramp, (dist_shift-dist_min_w_prev_ramp)/2, (dist_shift-dist_min_w_prev_ramp)/18)
                w_prev = linear_interp(x_prev, 1, 0, w_forceRef_big_newcontact, conf.w_forceRef)
                tsid_solo.contacts[prev_raised_foot_nb].setRegularizationTaskWeightVector(w_prev*np.ones(3))

            if dist_to_goal <= dist_max_w_next_ramp:
                #
                # w_next = linear_interp(dist_to_goal, dist_max_w_next_ramp, 0, conf.w_forceRef, w_forceRef_big)
                #
                x_next = logistic.cdf(dist_max_w_next_ramp-dist_to_goal, dist_max_w_next_ramp/2, dist_max_w_next_ramp/18)
                w_next = linear_interp(x_next, 1, 0, w_forceRef_big, conf.w_forceRef)
                tsid_solo.contacts[raised_foot_nb].setRegularizationTaskWeightVector(w_next*np.ones(3))

            i_shift += 1
        
        else:
            print('\n\n\n\nfull support else')
            # Just in case
            # tsid_solo.contacts[prev_raised_foot_nb].setRegularizationTaskWeightVector(conf.w_forceRef*np.ones(3))
            # tsid_solo.contacts[raised_foot_nb].setRegularizationTaskWeightVector(conf.w_forceRef*np.ones(3))

            tsid_solo.remove_contact(raised_foot_nb)
            full_support = False
            partial_support = True
            t_partial = 0
    if partial_support:
        pos_f, vel_f, acc_f = compute_cos_traj(t, amp_lst[raised_foot_nb], freq)

        tsid_solo.set_foot_3d_ref(pos_f, vel_f, acc_f, raised_foot_nb)

        t_partial += dt
        if t_partial > PARTIAL_SUPPORT_DURATION:
            print('\n\n\n\npartial support if')
            tsid_solo.add_contact(raised_foot_nb)
            prev_raised_foot_nb = raised_foot_nb
            raised_foot_nb += 1

            partial_support = False
            new_shift = True
            full_support = True
            
            if raised_foot_nb > 3:
                end_traj = True


    w_prev_arr.append(w_prev)
    w_next_arr.append(w_next)
    x_prev_arr.append(x_prev)
    x_next_arr.append(x_next)

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
    # q, v = tsid_solo.integrate_dv(q, v, dv, dt)
    q, v = tsid_solo.integrate_dv_R3SO3(q, v, dv, dt)
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
logger.store_csv_trajs('solo_stamping', sep=' ', skip_free_flyer=True)
logger.store_mcapi_traj(tsid_solo, 'solo_stamping')

import matplotlib.pyplot as plt


# FORCE
plt.figure('solo stamping contact forces')
plt.title('solo stamping contact forces')
plt.subplot(4,1,1)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,0], label='{}_fx'.format(logger.data_log['contact_names'][0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,0], label='{}_fx'.format(logger.data_log['contact_names'][1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,0], label='{}_fx'.format(logger.data_log['contact_names'][2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,0], label='{}_fx'.format(logger.data_log['contact_names'][3]))
plt.legend()
plt.grid()
plt.subplot(4,1,2)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,1], label='{}_fy'.format(logger.data_log['contact_names'][0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,1], label='{}_fy'.format(logger.data_log['contact_names'][1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,1], label='{}_fy'.format(logger.data_log['contact_names'][2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,1], label='{}_fy'.format(logger.data_log['contact_names'][3]))
plt.legend()
plt.grid()
plt.subplot(4,1,3)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][3]))
plt.legend()
plt.subplot(4,1,4)
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,0], label='{}_contact'.format(logger.data_log['contact_names'][0]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,1], label='{}_contact'.format(logger.data_log['contact_names'][1]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,2], label='{}_contact'.format(logger.data_log['contact_names'][2]))
plt.plot(logger.data_log['t'], logger.data_log['contacts'][:,3], label='{}_contact'.format(logger.data_log['contact_names'][3]))
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


plt.figure('solo stamping configuration position')
plt.title('solo stamping configuration position')
plt.subplot(4,1,1)
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*0+0], label='{}_q'.format(7+3*0+0))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*0+1], label='{}_q'.format(7+3*0+1))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*0+2], label='{}_q'.format(7+3*0+2))
plt.legend()
plt.grid()
plt.subplot(4,1,2)
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*1+0], label='{}_q'.format(7+3*1+0))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*1+1], label='{}_q'.format(7+3*1+1))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*1+2], label='{}_q'.format(7+3*1+2))
plt.legend()
plt.grid()
plt.subplot(4,1,3)
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*2+0], label='{}_q'.format(7+3*2+0))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*2+1], label='{}_q'.format(7+3*2+1))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*2+2], label='{}_q'.format(7+3*2+2))
plt.legend()
plt.subplot(4,1,4)
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*3+0], label='{}_q'.format(7+3*3+0))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*3+1], label='{}_q'.format(7+3*3+1))
plt.plot(logger.data_log['t'], logger.data_log['q'][:,3*3+2], label='{}_q'.format(7+3*3+2))
plt.legend()
plt.grid()

plt.figure('solo stamping configuration velocity')
plt.title('solo stamping configuration velocity')
plt.subplot(4,1,1)
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*0+0], label='{}_v'.format(6+3*0+0))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*0+1], label='{}_v'.format(6+3*0+1))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*0+2], label='{}_v'.format(6+3*0+2))
plt.legend()
plt.grid()
plt.subplot(4,1,2)
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*1+0], label='{}_v'.format(6+3*1+0))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*1+1], label='{}_v'.format(6+3*1+1))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*1+2], label='{}_v'.format(6+3*1+2))
plt.legend()
plt.grid()
plt.subplot(4,1,3)
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*2+0], label='{}_v'.format(6+3*2+0))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*2+1], label='{}_v'.format(6+3*2+1))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*2+2], label='{}_v'.format(6+3*2+2))
plt.legend()
plt.subplot(4,1,4)
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*3+0], label='{}_v'.format(6+3*3+0))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*3+1], label='{}_v'.format(6+3*3+1))
plt.plot(logger.data_log['t'], logger.data_log['v'][:,3*3+2], label='{}_v'.format(6+3*3+2))
plt.legend()
plt.grid()

w_prev_arr = np.array(w_prev_arr)
w_next_arr = np.array(w_next_arr)
x_prev_arr = np.array(x_prev_arr)
x_next_arr = np.array(x_next_arr)

plt.figure()
plt.subplot(2,1,1)
plt.plot(logger.data_log['t'], w_prev_arr, label='w_prev')
plt.plot(logger.data_log['t'], w_next_arr, label='w_next')
plt.subplot(2,1,2)
plt.plot(logger.data_log['t'], logger.data_log['f0'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][0]))
plt.plot(logger.data_log['t'], logger.data_log['f1'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][1]))
plt.plot(logger.data_log['t'], logger.data_log['f2'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][2]))
plt.plot(logger.data_log['t'], logger.data_log['f3'][:,2], label='{}_fz'.format(logger.data_log['contact_names'][3]))
plt.legend()


plt.figure()
plt.plot(logger.data_log['t'], x_prev_arr, label='x_prev')
plt.plot(logger.data_log['t'], x_next_arr, label='x_next')
plt.legend()


plt.show()