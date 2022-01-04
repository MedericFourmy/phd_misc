import numpy as np
import pinocchio as pin
from scipy import optimize
import matplotlib.pyplot as plt



class CostAvgSE3:
    def __init__(self, Tm_lst):
        self.Tm_lst = Tm_lst
        self.T_inv_lst = [T.inverse() for T in Tm_lst]
        self.N = len(Tm_lst)
        self.N_res = self.N*6  # se3

    def f(self, x):
        res = np.zeros(self.N_res)
        # nu = x  # se3 parameterization of SE3 transform to estimate
        T = pin.exp(x)

        for k, Tm_inv in enumerate(self.T_inv_lst):
            res[6*k:6*k+6] = pin.log(T*Tm_inv).vector

        return res
    
    def sol(self, r):
        return pin.exp(r.x)
    
    def plot_residuals(self, r):
        res = r.fun.reshape((self.N, 6))

        t_arr = np.arange(self.N)
        plt.figure('RES')
        plt.subplot(1,2,1)
        plt.title('se3 T err')
        for i in range(3):
            plt.plot(t_arr, res[:,i])
        plt.subplot(1,2,2)
        plt.title('se3 O err')
        for i in range(3,6):
            plt.plot(t_arr, res[:,i])
        plt.legend()




# Simulate data
N = 50
R = pin.rpy.rpyToMatrix(np.deg2rad(np.array([95,100,105]))) 
t = np.array([3,3,2])
T = pin.SE3(R, t)
A = 0.1
Tm_lst = [T*pin.exp(A*np.random.normal(np.zeros(6), A*np.ones(6))) for _ in range(N)]

def scipy_opt(Tm_lst):
    cost = CostAvgSE3(Tm_lst)
    x0 = np.zeros(6)

    r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', loss='huber', verbose=2, ftol=1e-6)
    cost.plot_residuals(r)

    return cost.sol(r)

def se3_avg(Tm_lst):
    nu_sum = sum([pin.log(Tm).vector for Tm in Tm_lst])
    return pin.exp(nu_sum/len(Tm_lst))

T_scipy = scipy_opt(Tm_lst)
T_avg = se3_avg(Tm_lst)
print('t_scipy: ', T_scipy.translation)
print('rpy_scipy: ', np.rad2deg(pin.rpy.matrixToRpy(T_scipy.rotation)))
print('t_avg: ', T_scipy.translation)
print('rpy_avg: ', np.rad2deg(pin.rpy.matrixToRpy(T_avg.rotation)))
T_diff = T_scipy*T_avg.inverse()
print()
print('t_diff: ', T_diff.translation)
print('rpy_diff: ', np.rad2deg(pin.rpy.matrixToRpy(T_diff.rotation)))

plt.show()