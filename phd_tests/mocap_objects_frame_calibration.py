import numpy as np
import pinocchio as pin
from scipy import optimize


class CostFrameCalibration:
    def __init__(self, bm_M_cm_traj, c_M_b_cosy_traj, cm_M_c0, bm_M_b0):
        self.x_arr = []
        self.cost_arr = []
        self.bm_M_cm_traj = bm_M_cm_traj
        self.c_M_b_cosy_traj = c_M_b_cosy_traj
        self.cm_M_c0 = cm_M_c0  # prior on constant transformations
        self.bm_M_b0 = bm_M_b0  # prior on constant transformations
        self.N = len(m_M_bm_traj)
        self.N_res = N*6  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        nu_c = x[:6]  # currently estimated transfo as se3 6D vector representation
        nu_b = x[6:]  # currently estimated transfo as se3 6D vector representation
        cm_M_c = self.cm_M_c0*pin.exp6(nu_c)  # currently estimated transfo as SE3 Lie group
        bm_M_b = self.cm_M_c0*pin.exp6(nu_b)  # currently estimated transfo as SE3 Lie group
        b_M_bm = bm_M_b.inverse()
        res = np.zeros(self.N_res)
        for i in range(self.N):
            bm_M_cm = self.bm_M_cm_traj[i]
            c_M_b  = self.c_M_b_cosy_traj[i]
            res[6*i:6*i+6] = pin.log6(bm_M_cm * cm_M_c * c_M_b * b_M_bm).np

        self.cost_arr.append(np.linalg.norm(res))

        return res

if __name__ == '__main__':
    N = 1000
    dt = 0.01
    print('Tot seconds ', N*dt)

    t = np.arange(N)

    A = 0.5*(np.random.random(6) - 0.5)
    f = 0.2*(np.random.random(6) - 0.5)
    ft_arr = np.outer(t, f)
    nu_traj = A*np.sin(2*np.pi*ft_arr)
    # nu_traj = A*(np.random.random((N,6)) - 0.5)


    m_M_c_traj = [pin.exp6(nu) for nu in nu_traj]  # moving camera
    m_M_b_traj = [pin.SE3.Identity() for _ in range(N)]   # resting object

    # cozy pose measurements
    c_M_b_cosy_traj = [m_M_c.inverse()*m_M_b for m_M_c, m_M_b in zip(m_M_c_traj, m_M_b_traj)]

    # constant transformation to estimate
    cm_M_c = pin.SE3.Random()
    bm_M_b = pin.SE3.Random()
    c_M_cm = cm_M_c.inverse()
    b_M_bm = bm_M_b.inverse()

    # mocap measurements
    m_M_cm_traj = [m_M_c*c_M_cm for m_M_c in m_M_c_traj]
    m_M_bm_traj = [m_M_b*b_M_bm for m_M_b in m_M_b_traj]
    # we only need the relative transformation for computations
    bm_M_cm_traj = [m_M_bm.inverse() * m_M_cm for m_M_cm, m_M_bm in zip(m_M_cm_traj, m_M_bm_traj)]
   
    # priors
    cm_M_c0 = pin.SE3.Identity()
    bm_M_b0 = pin.SE3.Identity()
    cost = CostFrameCalibration(bm_M_cm_traj, c_M_b_cosy_traj, cm_M_c0, bm_M_b0)
    

    x0 = np.zeros(12)  # chosen to be [nu_c, nu_b]
    r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
    # r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', verbose=2)
    # r = optimize.least_squares(cost.f, x0, jac='3-point', method='trf', loss='huber', verbose=2)
    # r = optimize.least_squares(cost.f, x0, jac=cost.jac, method='trf', loss='huber', verbose=2)

    nu_c = r.x[:6]
    nu_b = r.x[6:]

    # recover estimated cst transfo
    cm_M_c_est = cm_M_c0*pin.exp6(nu_c) 
    bm_M_b_est = bm_M_b0*pin.exp6(nu_b)

    
    print('cm_M_c_est\n', cm_M_c_est)
    print('bm_M_b_est\n', bm_M_b_est)
    
    print('cm_M_c gtr\n', cm_M_c)
    print('bm_M_b gtr\n', bm_M_b)

    print('cm_M_c_err', pin.log6(cm_M_c_est.inverse()*cm_M_c))
    print('bm_M_b_err', pin.log6(bm_M_b_est.inverse()*bm_M_b))
    print()
    print('r.cost')
    print(r.cost)
    print()

    # residuals RMSE
    res = r.fun.reshape((N, 6))
    rmse_arr = np.sqrt(np.mean(res**2, axis=0))

    # examine the problem jacobian at the solution
    J = r.jac
    H = J.T @ J
    u, s, vh = np.linalg.svd(H, full_matrices=True)

    import matplotlib.pyplot as plt

    plt.figure('cost evolution')
    plt.plot(np.arange(len(cost.cost_arr)), np.log(cost.cost_arr))
    plt.xlabel('Iterations')
    plt.ylabel('Residuals norm')

    plt.figure('Hessian singular values')
    plt.bar(np.arange(len(s)), np.log(s))
    plt.xlabel('degrees of freedom')
    plt.ylabel('log(s)')

    print('Hessian singular values')
    print(s)


    plt.show()
