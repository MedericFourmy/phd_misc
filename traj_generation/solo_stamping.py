import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_quadruped import TsidQuadruped
import conf_solo12 as conf

dt = conf.dt

tsid_solo = TsidQuadruped(conf, viewer=True)

data = tsid_solo.invdyn.data()
robot = tsid_solo.robot

# Params for Com trajectory
SHIFT_DURATION = 3
pos_f0, _, _ = tsid_solo.get_3d_pos_vel_acc(np.zeros(3), 0)
pos_f3, _, _ = tsid_solo.get_3d_pos_vel_acc(np.zeros(3), 2)
pos_com_init = robot.com(data)
pos_c = pos_com_init.copy()
shift_traj = np.linspace(pos_c[:2], pos_f0[:2], SHIFT_DURATION/dt)
# print(shift_traj)

shift_traj[1000:] = shift_traj[1000] 

print(shift_traj.shape)

# param for RF traj during "swing phase"
amp        = np.array([0.01, 0.0, -0.02])                    # amplitude function
# amp        = np.array([0.0, 0.0, 0.0])                    # amplitude function
offset     = pos_f3 - amp                          
two_pi_f             = 2*np.pi*np.array([0.2, 0.2, 0.2])   # movement frequencies along each axis
two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function


# Init values
q, v = tsid_solo.q, tsid_solo.v
t = 0.0 # time
time_start = time.time()
for i in range(0, conf.N_SIMULATION):

    # simple switch
    if i < shift_traj.shape[0]:
        pos_c[:2] = shift_traj[i]
    elif tsid_solo.active_contacts[3]:
        tsid_solo.remove_contact(3)
    else:
        pos_RF = offset + amp * np.cos(two_pi_f*(t-SHIFT_DURATION))
        vel_RF = two_pi_f_amp * (-np.sin(two_pi_f*(t-SHIFT_DURATION)))
        acc_RF = two_pi_f_squared_amp * (-np.cos(two_pi_f*(t-SHIFT_DURATION)))
        tsid_solo.set_foot_3d_ref(pos_RF, vel_RF, acc_RF, 3)
    
    # dummy values, the com position trajectory is linear (bad)
    # hence the control will lag behind. This is just for testing.
    vel_c = np.zeros(3)
    acc_c = np.zeros(3)    
    tsid_solo.set_com_ref(pos_c, vel_c, acc_c)

    # Solve
    sol, HQdata = tsid_solo.compute_and_solve(t, q, v)

    # Store values
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    tau = tsid_solo.invdyn.getActuatorForces(sol)
    dv  = tsid_solo.invdyn.getAccelerations(sol)

    # integrate one step
    q, v = tsid_solo.integrate_dv(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_solo.print_solve_check(sol, t, v, dv) 

    if (i % conf.DISPLAY_N) == 0:
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_solo.update_display(q, t)
        time_start = time.time()



