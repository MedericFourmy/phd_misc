#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pinocchio as pin

gravity = np.array((0,0,-9.81))



def compute_current_delta_IMU(b_ab, b_wb, dt):
    return [
        0.5 * b_ab * dt**2,
        b_ab * dt,
        pin.exp3(b_wb * dt)
    ]

def compute_current_delta_FT(fl1, fl2, taul1, taul2, b_pbc, b_wb, pbl1, bRl1, pbl2, bRl2):
    return np.array([
        # 0.5 * b_ab * dt**2,
        # b_ab * dt,
        # pin.exp3(b_wb * dt)
    ])

def compose_delta_IMU(Delta1, Delta2, dt):
    return [
        Delta1[0] + Delta1[1]*dt + Delta1[2] @ Delta2[0],
        Delta1[1] + Delta1[2] @ Delta2[1],
        Delta1[2] @ Delta2[2]
    ]

def compose_delta_FT(Delta1, Delta2):
    pass

def state_plus_delta_IMU(x, Delta, Deltat):
    return [
        x[0] + x[1]*Deltat + 0.5*gravity*Deltat**2 + x[2] @ Delta[0],
        x[1] + gravity*Deltat + x[2] @ Delta[1],
        x[2] @ Delta[2]
    ]


def quatarr_to_rot(quat_arr):
    return pin.Quaternion(quat_arr.reshape((4,1))).toRotationMatrix()

def imu_meas(dq, ddq, oRb):
    b_v = dq[0:3]
    b_w = dq[3:6]
    b_acc = ddq[0:3] + np.cross(b_w, b_v)
    b_proper_acc = b_acc - oRb.T @ gravity
    return b_w, b_acc, b_proper_acc


if __name__ == '__main__':
    # Tests
    N_SIMU = 10000
    dt = 1e-3
    t_arr = np.arange(0,N_SIMU*dt,dt)
    oRb = np.eye(3)
    p_int, v_int, oRb_int = np.zeros(3), np.zeros(3), np.eye(3)
    x_imu_ori = np.zeros(3), np.zeros(3), np.eye(3)
    x_imu_int = np.zeros(3), np.zeros(3), np.eye(3)
    DeltaIMU = np.zeros(3), np.zeros(3), np.eye(3)
    Deltat = 0

    v_direct_lst = [] 
    v_preint_lst = [] 
    for i in range(N_SIMU):
        v_direct_lst.append(v_int.copy())
        v_preint_lst.append(x_imu_int[1].copy())

        b_a = np.random.random(3) - 0.5
        b_w = np.random.random(3) - 0.5
        b_proper_acc = b_a - oRb_int.T @ gravity

        # direct preint
        p_int = p_int + v_int*dt  + 0.5*oRb_int @ b_a*dt**2
        v_int = v_int + oRb_int @ b_a*dt
        oRb_int = oRb_int @ pin.exp(b_w*dt)

        # preint
        Deltat += dt
        deltak = compute_current_delta_IMU(b_proper_acc, b_w, dt)
        DeltaIMU = compose_delta_IMU(DeltaIMU, deltak, dt)
        x_imu_int = state_plus_delta_IMU(x_imu_ori, DeltaIMU, Deltat)

        # pin
        b_nu_int += pin.Motion(b_dnu * dt)




    import matplotlib.pyplot as plt
    
    v_direct_arr = np.array(v_direct_lst)
    v_preint_arr = np.array(v_preint_lst)
    v_err = v_direct_arr - v_preint_arr
  
    plt.figure('Velocity direct - preint (error)')
    plt.plot(t_arr, v_err[:,0], 'r')
    plt.plot(t_arr, v_err[:,1], 'g')
    plt.plot(t_arr, v_err[:,2], 'b')
    plt.show()
