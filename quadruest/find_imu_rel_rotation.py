import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

# How to solve the orthogonal Procrustes problem
# https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
# Find: bRa* = argmin ||bRa*A - B||^2

f_sum_arr = np.load('f_sum_arr.npy')
N = f_sum_arr.shape[0]
f_sum_arr = f_sum_arr.T  # need to have shape (3,N)
m_solo = 2.5
g = 9.81
mg_arr = np.repeat(m_solo*g*np.array([0,0,1]).reshape(3,1), N, axis=1)


S = 2000
E = 10000

# print(f_sum_arr[:,S:E])
# print()
# print(mg_arr[:,S:E])
M = f_sum_arr[:,S:E] @ mg_arr[:,S:E].T
u, s, vh = np.linalg.svd(M, full_matrices=True)

R_est = u @ vh

print(R_est)
q_est = pin.Quaternion(R_est)
print(q_est.coeffs())
rpy_est = pin.rpy.matrixToRpy(R_est)
print('Angular error (in deg): ', np.rad2deg(pin.log3(R_est)))

f_sum_arr_rot = R_est.T@f_sum_arr
# print(f_sum_arr_rot[:,S:E])

t_arr = np.arange(N)

plt.figure('Total forces world frame')
plt.plot(t_arr, f_sum_arr[0,:], 'r', label='f sum x')
plt.plot(t_arr, f_sum_arr[1,:], 'g', label='f sum y')
plt.plot(t_arr, f_sum_arr[2,:], 'b', label='f sum z')
plt.hlines(0, -1000, N+1000, 'k')
plt.hlines(m_solo*g, -1000, N+1000, 'k')
plt.legend()

plt.figure('Total rotated forces world frame')
plt.plot(t_arr, f_sum_arr_rot[0,:], 'r', label='f sum x')
plt.plot(t_arr, f_sum_arr_rot[1,:], 'g', label='f sum y')
plt.plot(t_arr, f_sum_arr_rot[2,:], 'b', label='f sum z')
plt.hlines(0, -1000, N+1000, 'k')
plt.hlines(m_solo*g, -1000, N+1000, 'k')
plt.legend()

plt.show()