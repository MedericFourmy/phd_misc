import numpy as np
import pinocchio as pin

# How to solve the orthogonal Procrustes problem
# https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
# Find: bRa* = argmin ||bRa*A - B||^2

# create some data
bRa = pin.SE3.Random().rotation

N = 1000
a_f = np.random.random((N,3))
a_f = a_f.T  # put in the right shape
b_f = bRa@a_f

M = b_f @ a_f.T
u, s, vh = np.linalg.svd(M, full_matrices=True)

bRa_est = u @ vh

assert(np.allclose(bRa, bRa_est))

# same but add some noise

a_f += np.random.normal(0,0.05,(3,N))
b_f += np.random.normal(0,0.05,(3,N))
M = b_f @ a_f.T
u, s, vh = np.linalg.svd(M, full_matrices=True)

bRa_est = u @ vh

err_o = pin.rpy.matrixToRpy(bRa_est@bRa.T)
print(np.rad2deg(err_o))