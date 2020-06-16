import time
import numpy as np
import pinocchio as pin
import tsid
from tsid_biped import TsidBiped
import conf_talos as conf


tsid_talos = TsidBiped(conf, viewer=True)

data = tsid_talos.invdyn.data()
robot = tsid_talos.robot

# Params for Com trajectory
amp        = np.array([0.0, 0.03, 0.00])                    # amplitude functio
# amp        = np.array([0.0, 0.0, 0.0])                    # amplitude functio
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
    # vel_c = np.zeros(3)
    # acc_c = np.zeros(3)
    vel_c = two_pi_f_amp * (-np.sin(two_pi_f*t))
    acc_c = two_pi_f_squared_amp * (-np.cos(two_pi_f*t))
    tsid_talos.set_com_ref(pos_c, vel_c, acc_c)

    # Solve
    sol, HQdata = tsid_talos.compute_and_solve(t, q, v)

    # Store values
    if(sol.status!=0):
        print ("QP problem could not be solved! Error code:", sol.status)
        break
    
    tau = tsid_talos.invdyn.getActuatorForces(sol)
    dv  = tsid_talos.invdyn.getAccelerations(sol)

    # integrate one step
    q, v = tsid_talos.integrate_dv(q, v, dv, dt)
    t += dt

    if (i % conf.PRINT_N) == 0:
        tsid_talos.print_solve_check(sol, t, v, dv) 

    if (i % conf.DISPLAY_N) == 0:
        HQdata.print_all()
        time_spent = time.time() - time_start
        if(time_spent < dt*conf.DISPLAY_N): time.sleep(dt*conf.DISPLAY_N-time_spent)
        tsid_talos.update_display(q, t)
        time_start = time.time()



