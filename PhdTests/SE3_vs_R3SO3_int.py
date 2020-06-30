
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

dt = 1e-3
N_SIMU = 10000
t_arr = np.arange(0,N_SIMU*dt,dt)

# initial state vel
b_nu = pin.Motion.Zero()
b_nu_int = b_nu.copy()

# SE3 int variables
oMb_int = pin.SE3.Identity()
p_int_se3 = np.zeros(3)
oRb_int_se3 = np.eye(3)
v_int_se3 = np.zeros(3)
# R3xSO3 int variables
p_int_so3 = np.zeros(3)
oRb_int_so3 = np.eye(3)
v_int_so3 = np.zeros(3)

p_se3_lst = []
oRb_se3_lst = []
v_se3_lst = []
p_so3_lst = []
oRb_so3_lst = []
v_so3_lst = []

dnu_lst = []
for i in range(N_SIMU):
    # time i
    p_se3_lst.append(p_int_se3.copy())
    oRb_se3_lst.append(oRb_int_se3.copy())
    v_se3_lst.append(v_int_se3.copy())
    p_so3_lst.append(p_int_so3.copy())
    oRb_so3_lst.append(oRb_int_so3.copy())
    v_so3_lst.append(v_int_so3.copy())

    # time i + 1
    # generate rand input data
    b_dnu = 1e-3*(pin.utils.rand(6) - 0.5)
    b_dnu[3:6] = 1e-3

    # velocity one step integration
    b_nu_int += pin.Motion(b_dnu * dt)
    b_v = b_nu_int.linear
    b_w = b_nu_int.angular
    b_acc = b_dnu[0:3] + np.cross(b_w, b_v)

    # SE3 integration
    cur_M_next = pin.exp6(b_nu_int * dt)
    oMb_int = oMb_int * cur_M_next
    p_int_se3 = oMb_int.translation
    oRb_int_se3 = oMb_int.rotation
    v_int_se3 = oRb_int_se3@b_nu_int.linear

    # R3xSO3 integration
    v_int_so3 += oRb_int_so3 @ b_acc * dt
    oRb_int_so3 = oRb_int_so3 @ pin.exp3(b_w*dt)
    p_int_so3 += v_int_so3*dt

    dnu_lst.append(b_dnu)


p_se3_arr = np.array(p_se3_lst)
v_se3_arr = np.array(v_se3_lst)
o_se3_arr = np.array([pin.log3(oRb_se3) for oRb_se3 in oRb_se3_lst])

p_so3_arr = np.array(p_so3_lst)
v_so3_arr = np.array(v_so3_lst)
o_so3_arr = np.array([pin.log3(oRb_so3) for oRb_so3 in oRb_so3_lst])

p_err_arr = p_so3_arr - p_se3_arr
v_err_arr = v_so3_arr - v_se3_arr
oRb_err_arr = np.array([pin.log3(oRb_est @ oRb_gtr.T) for oRb_est, oRb_gtr in zip(oRb_so3_lst, oRb_se3_lst)])

dnu_arr = np.array(dnu_lst)

# errors
err_arr_lst = [p_err_arr, oRb_err_arr, v_err_arr]
fig_titles = ['P error', 'O error', 'V error']
for err_arr, fig_title in zip(err_arr_lst, fig_titles):
    plt.figure(fig_title)
    for axis, (axis_name, c) in enumerate(zip('xyz', 'rgb')):
        plt.plot(t_arr, err_arr[:,axis], c, label='err_'+axis_name)
    plt.legend()
    plt.title(fig_title)

# ground truth vs est
so3_arr_lst = [p_so3_arr, o_so3_arr, v_so3_arr]
se3_arr_lst = [p_se3_arr, o_se3_arr, v_se3_arr]
fig_titles = ['P se3 vs so3', 'O se3 vs so3', 'V se3 vs so3']
for so3_arr, se3_arr, fig_title in zip(so3_arr_lst, se3_arr_lst, fig_titles):
    plt.figure(fig_title)
    for axis, (axis_name, c) in enumerate(zip('xyz', 'rgb')):
        plt.plot(t_arr, so3_arr[:,axis], c+':', label='so3_'+axis_name)
        plt.plot(t_arr, se3_arr[:,axis], c, label='se3_'+axis_name)
    plt.legend()
    plt.title(fig_title)


# plt.figure()
# for i in range(6):
#     plt.plot(t_arr, dnu_arr[:,i], label=str(i))
# plt.legend()

plt.show()
