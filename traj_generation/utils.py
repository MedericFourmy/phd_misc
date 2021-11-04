import numpy as np

def compute_other_feet(fnb, feet_nb):
    return [nb for nb in feet_nb if fnb != nb]

def linear_interp(x, xa, xb, ya, yb):
    return ya + (x-xa)*(yb-ya)/(xb-xa)

def dist(posa, posb):
    return np.linalg.norm(posa-posb)

def compute_cos_traj(t, amp, offset, freq):
    # param for RF traj during "swing phase"
    two_pi_f             = 2*np.pi*freq   # movement frequencies along each axis
    two_pi_f_amp         = two_pi_f * amp                      # 2π function times amplitude function
    two_pi_f_squared_amp = two_pi_f * two_pi_f_amp             # 2π function times squared amplitude function
    pos = offset + amp * np.cos(two_pi_f*(t))
    vel = two_pi_f_amp * (-np.sin(two_pi_f*(t)))
    acc = two_pi_f_squared_amp * (-np.cos(two_pi_f*(t)))
    return pos, vel, acc
