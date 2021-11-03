import json
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from collections.abc import Iterable

from scipy.ndimage.measurements import label


WEIGHT_REGU = 1e-5




class CostOffsetDistances:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        self.combinations = [
            (0,1),
            (0,2),
            (0,3),
            (1,2),
            (1,3),
            (2,3),
        ]
        self.dists_gtr = [np.linalg.norm(self.ω_p_ωl_arr[comb[1]] - self.ω_p_ωl_arr[comb[0]]) 
                             for comb in self.combinations]
        self.N_comb = len(self.combinations)
        self.comb_ids = [(self.cids[c[0]], self.cids[c[1]]) for c in self.combinations]
        self.comb_names = [(self.LEGS[c[0]]+self.LEGS[c[1]]) for c in self.combinations]

        self.N_res = self.N*self.N_comb
        # self.N_res = self.N*(self.N_comb + 1)  # coplanarity 



    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        offset = x[:12]
        dists = x[12:]

        for k, qa in enumerate(self.qa_arr):
            q[7:] = qa + offset
            self.robot.forwardKinematics(q)
            for ic, cid in enumerate(self.comb_ids):
                pa = self.robot.framePlacement(q, cid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, cid[1], update_kinematics=True).translation
                res[self.N_comb*k + ic] = np.linalg.norm(pb - pa) - dists[ic]

            # # coplanarity
            # res[self.N_comb*k+self.N_comb] 


        return res
    
    def jac(self, x):
        offset = x[:12]
        J = np.zeros((self.N*self.N_comb,18))
        q = self.robot.q0.copy()

        for k, qa in enumerate(self.qa_arr):
            q[7:] = qa + offset
            self.robot.forwardKinematics(q)
            for ic, cid in enumerate(self.comb_ids):
                pa = self.robot.framePlacement(q, cid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, cid[1], update_kinematics=True).translation
                v = pb - pa
                self.robot.computeJointJacobians(q)
                self.robot.framesForwardKinematics(q)
                # take only the jacobian wrt. the actuated part
                oJa = self.robot.getFrameJacobian(cid[0], rf_frame=pin.LOCAL_WORLD_ALIGNED)[:3, 6:]
                oJb = self.robot.getFrameJacobian(cid[1], rf_frame=pin.LOCAL_WORLD_ALIGNED)[:3, 6:]
                
                # jacobian wrt. joint deltas
                J[self.N_comb*k + ic,:12] = v.T@(oJb - oJa) / np.linalg.norm(v)
                # jacobian wrt. combination distances 
                J[self.N_comb*k + ic,12+ic] = -1

        return J

    def plot_residuals(self, r):
        res = r.fun.reshape((self.N, 6))

        t_arr = np.arange(self.N)
        plt.figure('Res distances')
        for i in range(6):
            plt.plot(t_arr, res[:,i], label=self.comb_names[i])
        plt.legend()


    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        offset_sample_arr = np.zeros((nsol,12))
        distance_sample_arr = np.zeros((nsol,6))
        
        for i, r in enumerate(r_lst):
            offset = r.x[:12]
            dists = r.x[12:]

            offset_sample_arr[i,:] = offset
            distance_sample_arr[i,:] = dists

        plt.figure('offsets')
        plt.title('offsets (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), offset_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('distances')
        plt.title('distances (m)')
        plt.plot(np.array(self.comb_names), self.dists_gtr, 'gx', markersize=10)
        plt.plot(np.array(self.comb_names), distance_sample_arr.T, '.', markersize=5)

        offset_mean = np.mean(offset_sample_arr, axis=0)
        dists_mean = np.mean(distance_sample_arr, axis=0)

        print('offset_mean', offset_mean)
        print('dists_mean', dists_mean)


    def save_calib(self, r, name='calib'):
        offset = r.x[:12]
        dists = r.x[12:]

        d = {
            'offset': offset.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')

    

# def check_finite_difference(cost):
#     x = np.random.random(18)
#     eps = 1e-10

#     Jan = cost.jac(x)

#     Jfin = np.zeros(Jan.shape)

#     for i in range(18):
#         dx = np.zeros(18)
#         dx[i] = eps
#         y = cost.f(x)
#         ydx = cost.f(x+dx)
#         Jfin[:,i] = (ydx - y)/eps

#     assert(np.allclose(Jan, Jfin, atol=1e-5))






class CostOffsetDistancesGtr:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        self.combinations = [
            (0,1),
            (0,2),
            (0,3),
            (1,2),
            (1,3),
            (2,3),
        ]
        self.dists_gtr = [np.linalg.norm(self.ω_p_ωl_arr[comb[1]] - self.ω_p_ωl_arr[comb[0]]) 
                             for comb in self.combinations]
        self.N_comb = len(self.combinations)
        self.comb_ids = [(self.cids[c[0]], self.cids[c[1]]) for c in self.combinations]
        self.comb_names = [(self.LEGS[c[0]]+self.LEGS[c[1]]) for c in self.combinations]

        self.N_res = self.N*self.N_comb


    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        offset = x[:12]

        for k, qa in enumerate(self.qa_arr):
            q[7:] = qa + offset
            self.robot.forwardKinematics(q)
            for ic, cid in enumerate(self.comb_ids):
                pa = self.robot.framePlacement(q, cid[0], update_kinematics=True).translation
                pb = self.robot.framePlacement(q, cid[1], update_kinematics=True).translation
                res[self.N_comb*k + ic] = np.linalg.norm(pb - pa) - self.dists_gtr[ic]

        return res
    

    def plot_residuals(self, r):
        res = r.fun.reshape((self.N, 6))

        t_arr = np.arange(self.N)
        plt.figure('Res distances')
        for i in range(6):
            plt.plot(t_arr, res[:,i], label=self.comb_names[i])
        plt.legend()


    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        offset_sample_arr = np.zeros((nsol,12))
        
        for i, r in enumerate(r_lst):
            offset = r.x[:12]

            offset_sample_arr[i,:] = offset

        plt.figure('offsets')
        plt.title('offsets (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), offset_sample_arr[:,i::3].T, '.', markersize=3)

        offset_mean = np.mean(offset_sample_arr, axis=0)

        print('offset_mean', offset_mean)


    def save_calib(self, r, name='calib'):
        offset = r.x[:12]

        d = {
            'offset': offset.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')

  




class CostOffsetNew:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.tau_arr = params['tau_arr']
        self.w_p_wm_arr = params['w_p_wm_arr']
        self.w_q_m_arr = params['w_q_m_arr']
        self.m_M_b_init = params['m_M_b_init']
        self.w_M_ω_init = params['w_M_ω_init']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.height = params['height']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        self.N_res = self.N*4*3 + 1 + 1

    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        offset = x[:12]
        nu_b = x[12:18]
        nu_ω = x[-6:]

        # compute current extrinsics poses
        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        # wooden planck on the ground
        # res[-2] = w_M_ω.translation[2] - self.height
        # w_M_ω.translation[2] = 18e-3

        # compute feet positions in world frame
        w_p_wl_lst = [ w_M_ω * ω_p_ωl for ω_p_ωl in self.ω_p_ωl_arr] 

        # [print(toto.shape) for toto in [self.qa_arr, self.tau_arr, self.w_p_wm_arr, self.w_q_m_arr]]
        for k, (qa, tau, w_p_wm, w_q_m) in enumerate(zip(self.qa_arr, self.tau_arr, self.w_p_wm_arr, self.w_q_m_arr)):
            w_M_m = pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m]))
            w_M_b = w_M_m*m_M_b
            q[:7] = pin.SE3ToXYZQUAT(w_M_b)
            q[7:] = qa + offset
            self.robot.forwardKinematics(q)

            for i, cid in enumerate(self.cids):
                pi = self.robot.framePlacement(q, cid, update_kinematics=True).translation
                w_p_wl = w_p_wl_lst[i]
                res[4*3*k+3*i : 4*3*k+3*i + 3] = pi - w_p_wl

        # res[-1] = np.sum(nu_b**2)
        # res[-1] = np.sum((nu_b+nu_ω)**2)
        res[-1] = WEIGHT_REGU*np.sum(x**2)

        return res

    def plot_residuals(self, r):
        # -2 to take into account the reguralization costs
        res = r.fun[:-2].reshape((self.N, 12))

        t_arr = np.arange(self.N)
        plt.figure('Feet residuals')
        for ic in range(4):
            err = np.linalg.norm(res[:,3*ic:3*ic+3], axis=1)
            plt.plot(t_arr, err, '.', label=self.LEGS[ic], markersize=3)
        plt.xlabel('Ordered samples')
        plt.ylabel('Feet residuals norm (m) for each foot')
        plt.legend()

    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        offset_sample_arr = np.zeros((nsol,12))
        m_pose_b_sample_arr = np.zeros((nsol,6))
        w_pose_ω_sample_arr = np.zeros((nsol,6))
        
        for i, r in enumerate(r_lst):
            offset = r.x[:12]
            nu_b = r.x[12:18]
            nu_ω = r.x[-6:]

            offset_sample_arr[i,:] = offset
            m_M_b = self.m_M_b_init*pin.exp(nu_b)
            w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
            m_pose_b_sample_arr[i,:3] = m_M_b.translation
            m_pose_b_sample_arr[i,3:] = nu_b[3:]
            w_pose_ω_sample_arr[i,:3] = w_M_ω.translation
            w_pose_ω_sample_arr[i,3:] = nu_ω[3:]

        plt.figure('offsets')
        plt.title('offsets (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), offset_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('m_M_b')
        plt.subplot(1,2,1)
        plt.title('m_M_b trans (m)')
        plt.plot(['tx', 'ty', 'tz'], m_pose_b_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('m_M_b rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(m_pose_b_sample_arr[:,3:].T), '.', markersize=3)

        plt.figure('w_M_ω')
        plt.subplot(1,2,1)
        plt.title('w_M_ω trans (m)')
        plt.plot(['tx', 'ty', 'tz'], w_pose_ω_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('w_M_ω rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(w_pose_ω_sample_arr[:,3:].T), '.', markersize=3)

        print('MEANS:')
        m_p_mb_mean = np.mean(m_pose_b_sample_arr[:,:3], axis=0)
        m_o_b_mean = np.mean(m_pose_b_sample_arr[:,3:], axis=0)
        m_quat_b_mean = pin.Quaternion(pin.exp(m_o_b_mean)).coeffs()

        w_p_wω_mean = np.mean(w_pose_ω_sample_arr[:,:3], axis=0)
        w_o_ω_mean = np.mean(w_pose_ω_sample_arr[:,3:], axis=0)
        w_quat_ω_mean = pin.Quaternion(pin.exp(w_o_ω_mean)).coeffs()

        offset_mean = np.mean(offset_sample_arr, axis=0)

        print('m_p_mb_mean', m_p_mb_mean)
        print('m_quat_b_mean', m_quat_b_mean)
        print('w_p_wω_mean', w_p_wω_mean)
        print('w_quat_ω_mean', w_quat_ω_mean)
        print('offset_mean', offset_mean)


    def save_calib(self, r, name='calib'):
        offset = r.x[:12]
        nu_b = r.x[-12:-6]
        nu_ω = r.x[-6:]

        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        m_p_mb = m_M_b.translation
        m_q_b = pin.Quaternion(m_M_b.rotation).coeffs()
        w_p_wω = w_M_ω.translation
        w_q_ω = pin.Quaternion(w_M_ω.rotation).coeffs()

        d = {
            'offset': offset.tolist(), 
            'm_p_mb': m_p_mb.tolist(), 
            'm_q_b': m_q_b.tolist(), 
            'w_p_wω': w_p_wω.tolist(), 
            'w_q_ω': w_q_ω.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')




class CostFlexiNew:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.tau_arr = params['tau_arr']
        self.w_p_wm_arr = params['w_p_wm_arr']
        self.w_q_m_arr = params['w_q_m_arr']
        self.m_M_b_init = params['m_M_b_init']
        self.w_M_ω_init = params['w_M_ω_init']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.height = params['height']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        # compute size of the residual
        self.N_res = self.N*4*3 + 1 + 1

    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        alpha = x[:12]
        nu_b = x[12:18]
        nu_ω = x[-6:]

        # compute current extrinsics poses
        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        # wooden planck on the ground
        # res[-2] = w_M_ω.translation[2] - self.height
        # w_M_ω.translation[2] = 18e-3

        # compute feet positions in world frame
        w_p_wl_lst = [ w_M_ω * ω_p_ωl for ω_p_ωl in self.ω_p_ωl_arr] 

        # [print(toto.shape) for toto in [self.qa_arr, self.tau_arr, self.w_p_wm_arr, self.w_q_m_arr]]
        for k, (qa, tau, w_p_wm, w_q_m) in enumerate(zip(self.qa_arr, self.tau_arr, self.w_p_wm_arr, self.w_q_m_arr)):
            w_M_m = pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m]))
            w_M_b = w_M_m*m_M_b
            q[:7] = pin.SE3ToXYZQUAT(w_M_b)
            q[7:] = qa + alpha*tau
            self.robot.forwardKinematics(q)

            for i, cid in enumerate(self.cids):
                pi = self.robot.framePlacement(q, cid, update_kinematics=True).translation
                w_p_wl = w_p_wl_lst[i]
                res[4*3*k+3*i : 4*3*k+3*i + 3] = pi - w_p_wl

        # res[-1] = np.sum(nu_b**2)
        # res[-1] = np.sum((nu_b+nu_ω)**2)
        res[-1] = WEIGHT_REGU*np.sum(x**2)

        return res

    def plot_residuals(self, r):
        # -2 to take into account the reguralization costs
        res = r.fun[:-2].reshape((self.N, 12))

        t_arr = np.arange(self.N)
        plt.figure('Feet residuals')
        for ic in range(4):
            err = np.linalg.norm(res[:,3*ic:3*ic+3], axis=1)
            plt.plot(t_arr, err, '.', label=self.LEGS[ic], markersize=3)
        plt.xlabel('Ordered samples')
        plt.ylabel('Feet residuals norm (m) for each foot')
        plt.legend()

    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        alpha_sample_arr = np.zeros((nsol,12))
        m_pose_b_sample_arr = np.zeros((nsol,6))
        w_pose_ω_sample_arr = np.zeros((nsol,6))
        
        for i, r in enumerate(r_lst):
            alpha = r.x[:12]
            nu_b = r.x[12:18]
            nu_ω = r.x[-6:]

            alpha_sample_arr[i,:] = alpha
            m_M_b = self.m_M_b_init*pin.exp(nu_b)
            w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
            m_pose_b_sample_arr[i,:3] = m_M_b.translation
            m_pose_b_sample_arr[i,3:] = nu_b[3:]
            w_pose_ω_sample_arr[i,:3] = w_M_ω.translation
            w_pose_ω_sample_arr[i,3:] = nu_ω[3:]

        plt.figure('alphas')
        plt.title('alphas (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), alpha_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('m_M_b')
        plt.subplot(1,2,1)
        plt.title('m_M_b trans (m)')
        plt.plot(['tx', 'ty', 'tz'], m_pose_b_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('m_M_b rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(m_pose_b_sample_arr[:,3:].T), '.', markersize=3)

        plt.figure('w_M_ω')
        plt.subplot(1,2,1)
        plt.title('w_M_ω trans (m)')
        plt.plot(['tx', 'ty', 'tz'], w_pose_ω_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('w_M_ω rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(w_pose_ω_sample_arr[:,3:].T), '.', markersize=3)

        print('MEANS:')
        m_p_mb_mean = np.mean(m_pose_b_sample_arr[:,:3], axis=0)
        m_o_b_mean = np.mean(m_pose_b_sample_arr[:,3:], axis=0)
        m_quat_b_mean = pin.Quaternion(pin.exp(m_o_b_mean)).coeffs()

        w_p_wω_mean = np.mean(w_pose_ω_sample_arr[:,:3], axis=0)
        w_o_ω_mean = np.mean(w_pose_ω_sample_arr[:,3:], axis=0)
        w_quat_ω_mean = pin.Quaternion(pin.exp(w_o_ω_mean)).coeffs()

        alpha_mean = np.mean(alpha_sample_arr, axis=0)

        print('m_p_mb_mean', m_p_mb_mean)
        print('m_quat_b_mean', m_quat_b_mean)
        print('w_p_wω_mean', w_p_wω_mean)
        print('w_quat_ω_mean', w_quat_ω_mean)
        print('alpha_mean', alpha_mean)
    
    def save_calib(self, r, name='calib'):
        alpha = r.x[:12]
        nu_b = r.x[-12:-6]
        nu_ω = r.x[-6:]

        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        m_p_mb = m_M_b.translation
        m_q_b = pin.Quaternion(m_M_b.rotation).coeffs()
        w_p_wω = w_M_ω.translation
        w_q_ω = pin.Quaternion(w_M_ω.rotation).coeffs()

        d = {
            'alpha': alpha.tolist(), 
            'm_p_mb': m_p_mb.tolist(), 
            'm_q_b': m_q_b.tolist(), 
            'w_p_wω': w_p_wω.tolist(), 
            'w_q_ω': w_q_ω.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')

        

class CostFlexiOffsetNew:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.tau_arr = params['tau_arr']
        self.w_p_wm_arr = params['w_p_wm_arr']
        self.w_q_m_arr = params['w_q_m_arr']
        self.m_M_b_init = params['m_M_b_init']
        self.w_M_ω_init = params['w_M_ω_init']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.height = params['height']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        # compute size of the residual
        self.N_res = self.N*4*3 + 1 + 1

    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        alpha = x[:12]
        offset = x[24:36]
        nu_b = x[-12:-6]
        nu_ω = x[-6:]

        # compute current extrinsics poses
        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        # wooden planck on the ground
        # res[-2] = w_M_ω.translation[2] - self.height
        # w_M_ω.translation[2] = 18e-3

        # compute feet positions in world frame
        w_p_wl_lst = [ w_M_ω * ω_p_ωl for ω_p_ωl in self.ω_p_ωl_arr] 

        for k in range(self.N):
            qa = self.qa_arr[k,:]
            tau = self.tau_arr[k,:]
            w_p_wm = self.w_p_wm_arr[k,:] 
            w_q_m = self.w_q_m_arr[k,:]
            
            w_M_m = pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m]))
            w_M_b = w_M_m*m_M_b
            q[:7] = pin.SE3ToXYZQUAT(w_M_b)
            q[7:] = qa + offset + alpha*tau
            self.robot.forwardKinematics(q)

            for i, cid in enumerate(self.cids):
                w_p_wl_pin = self.robot.framePlacement(q, cid, update_kinematics=True).translation
                # res[4*3*k+3*i : 4*3*k+3*i + 3] = w_p_wl_pin - w_p_wl
                res[4*3*k+3*i : 4*3*k+3*i + 3] = w_M_ω*w_p_wl_pin - self.ω_p_ωl_arr[i]

        # res[-1] = np.sum(nu_b**2)
        # res[-1] = np.sum((nu_b+nu_ω)**2)
        res[-1] = WEIGHT_REGU*np.sum(x**2)

        return res

    def plot_residuals(self, r):
        # -2 to take into account the reguralization costs
        res = r.fun[:-2].reshape((self.N, 12))

        t_arr = np.arange(self.N)
        plt.figure('Feet residuals')
        for ic in range(4):
            err = np.linalg.norm(res[:,3*ic:3*ic+3], axis=1)
            plt.plot(t_arr, err, '.', label=self.LEGS[ic], markersize=3)
        plt.xlabel('Ordered samples')
        plt.ylabel('Feet residuals norm (m) for each foot')
        plt.legend()

    
    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        alpha_sample_arr = np.zeros((nsol,12))
        offset_sample_arr = np.zeros((nsol,12))
        m_pose_b_sample_arr = np.zeros((nsol,6))
        w_pose_ω_sample_arr = np.zeros((nsol,6))
        
        for i, r in enumerate(r_lst):
            alpha = r.x[:12]
            offset = r.x[12:24]
            nu_b = r.x[-12:-6]
            nu_ω = r.x[-6:]

            alpha_sample_arr[i,:] = alpha
            offset_sample_arr[i,:] = offset
            m_M_b = self.m_M_b_init*pin.exp(nu_b)
            w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
            m_pose_b_sample_arr[i,:3] = m_M_b.translation
            m_pose_b_sample_arr[i,3:] = nu_b[3:]
            w_pose_ω_sample_arr[i,:3] = w_M_ω.translation
            w_pose_ω_sample_arr[i,3:] = nu_ω[3:]

        plt.figure('alphas')
        plt.title('alphas (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), alpha_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('offset')
        plt.title('offset (rad)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), alpha_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('m_M_b')
        plt.subplot(1,2,1)
        plt.title('m_M_b trans (m)')
        plt.plot(['tx', 'ty', 'tz'], m_pose_b_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('m_M_b rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(m_pose_b_sample_arr[:,3:].T), '.', markersize=3)

        plt.figure('w_M_ω')
        plt.subplot(1,2,1)
        plt.title('w_M_ω trans (m)')
        plt.plot(['tx', 'ty', 'tz'], w_pose_ω_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('w_M_ω rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(w_pose_ω_sample_arr[:,3:].T), '.', markersize=3)

        print('MEANS:')
        m_p_mb_mean = np.mean(m_pose_b_sample_arr[:,:3], axis=0)
        m_o_b_mean = np.mean(m_pose_b_sample_arr[:,3:], axis=0)
        m_quat_b_mean = pin.Quaternion(pin.exp(m_o_b_mean)).coeffs()

        w_p_wω_mean = np.mean(w_pose_ω_sample_arr[:,:3], axis=0)
        w_o_ω_mean = np.mean(w_pose_ω_sample_arr[:,3:], axis=0)
        w_quat_ω_mean = pin.Quaternion(pin.exp(w_o_ω_mean)).coeffs()

        alpha_mean = np.mean(alpha_sample_arr, axis=0)

        print('m_p_mb_mean', m_p_mb_mean)
        print('m_quat_b_mean', m_quat_b_mean)
        print('w_p_wω_mean', w_p_wω_mean)
        print('w_quat_ω_mean', w_quat_ω_mean)
        print('alpha_mean', alpha_mean)
        print('alpha_mean', alpha_mean)

    def save_calib(self, r, name='calib'):
        alpha = r.x[:12]
        offset = r.x[12:24]
        nu_b = r.x[-12:-6]
        nu_ω = r.x[-6:]

        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        m_p_mb = m_M_b.translation
        m_q_b = pin.Quaternion(m_M_b.rotation).coeffs()
        w_p_wω = w_M_ω.translation
        w_q_ω = pin.Quaternion(w_M_ω.rotation).coeffs()

        d = {
            'alpha': alpha.tolist(), 
            'offset': offset.tolist(), 
            'm_p_mb': m_p_mb.tolist(), 
            'm_q_b': m_q_b.tolist(), 
            'w_p_wω': w_p_wω.tolist(), 
            'w_q_ω': w_q_ω.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')




class CostFlexiOffsetFrictionNew:
    def __init__(self, params):
        self.robot = params['robot']
        self.qa_arr = params['qa_arr']
        self.dqa_arr = params['dqa_arr']
        self.tau_arr = params['tau_arr']
        self.w_p_wm_arr = params['w_p_wm_arr']
        self.w_q_m_arr = params['w_q_m_arr']
        self.m_M_b_init = params['m_M_b_init']
        self.w_M_ω_init = params['w_M_ω_init']
        self.ω_p_ωl_arr = params['ω_p_ωl_arr']
        self.height = params['height']
        self.cids = params['cids']
        self.N = params['N']
        self.LEGS = params['LEGS']

        # compute size of the residual
        self.N_res = self.N*4*3 + 1 + 1

    def f(self, x):
        res = np.zeros(self.N_res)
        q = self.robot.q0.copy()
        alpha = x[:12]
        offset = x[12:24]
        friction = x[24:36]
        nu_b = x[-12:-6]
        nu_ω = x[-6:]

        # compute current extrinsics poses
        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        # wooden planck on the ground
        # res[-2] = w_M_ω.translation[2] - self.height
        # w_M_ω.translation[2] = 18e-3

        # compute feet positions in world frame
        w_p_wl_lst = [ w_M_ω * ω_p_ωl for ω_p_ωl in self.ω_p_ωl_arr] 

        for k in range(self.N):
            qa = self.qa_arr[k,:]
            dqa = self.dqa_arr[k,:]
            tau = self.tau_arr[k,:]
            w_p_wm = self.w_p_wm_arr[k,:] 
            w_q_m = self.w_q_m_arr[k,:]
            
            
            w_M_m = pin.XYZQUATToSE3(np.concatenate([w_p_wm, w_q_m]))
            w_M_b = w_M_m*m_M_b
            q[:7] = pin.SE3ToXYZQUAT(w_M_b)
            # estimate current friction to compensate current measurements
            # tau_eff = tau  # no compensation
            # tau_eff = tau - friction*dqa  # linear model
            tau_eff = tau - friction*np.sign(dqa)  # static friction model
            q[7:] = qa + offset + alpha*tau_eff
            self.robot.forwardKinematics(q)

            for i, cid in enumerate(self.cids):
                pi = self.robot.framePlacement(q, cid, update_kinematics=True).translation
                w_p_wl = w_p_wl_lst[i]
                res[4*3*k+3*i : 4*3*k+3*i + 3] = pi - w_p_wl

        # res[-1] = np.sum(nu_b**2)
        # res[-1] = np.sum((nu_b+nu_ω)**2)
        res[-1] = WEIGHT_REGU*np.sum(x**2)

        return res

    def plot_residuals(self, r):
        # -2 to take into account the reguralization costs
        res = r.fun[:-2].reshape((self.N, 12))

        t_arr = np.arange(self.N)
        plt.figure('Feet residuals')
        for ic in range(4):
            err = np.linalg.norm(res[:,3*ic:3*ic+3], axis=1)
            plt.plot(t_arr, err, '.', label=self.LEGS[ic], markersize=3)
        plt.xlabel('Ordered samples')
        plt.ylabel('Feet residuals norm (m) for each foot')
        plt.legend()

    def print_n_plot_solutions(self, r_lst):
        # enforce a list
        if not isinstance(r_lst, Iterable):
            r_lst = [r_lst]

        nsol = len(r_lst)
        alpha_sample_arr = np.zeros((nsol,12))
        friction_sample_arr = np.zeros((nsol,12))
        offset_sample_arr = np.zeros((nsol,12))
        m_pose_b_sample_arr = np.zeros((nsol,6))
        w_pose_ω_sample_arr = np.zeros((nsol,6))
        
        for i, r in enumerate(r_lst):
            alpha = r.x[:12]
            friction = r.x[12:24]
            offset = r.x[24:36]
            nu_b = r.x[-12:-6]
            nu_ω = r.x[-6:]

            alpha_sample_arr[i,:] = alpha
            friction_sample_arr[i,:] = friction
            offset_sample_arr[i,:] = offset
            m_M_b = self.m_M_b_init*pin.exp(nu_b)
            w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
            m_pose_b_sample_arr[i,:3] = m_M_b.translation
            m_pose_b_sample_arr[i,3:] = nu_b[3:]
            w_pose_ω_sample_arr[i,:3] = w_M_ω.translation
            w_pose_ω_sample_arr[i,3:] = nu_ω[3:]

        plt.figure('alphas')
        plt.title('alphas (rad/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), alpha_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('friction')
        plt.title('friction (rad.s-1/N.m)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), friction_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('offset')
        plt.title('offset (rad)')
        for i in range(3):
            plt.subplot(1,3,1+i)
            plt.title(['HAA', 'HFE', 'KFE'][i])
            plt.plot(np.array(self.LEGS), alpha_sample_arr[:,i::3].T, '.', markersize=3)

        plt.figure('m_M_b')
        plt.subplot(1,2,1)
        plt.title('m_M_b trans (m)')
        plt.plot(['tx', 'ty', 'tz'], m_pose_b_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('m_M_b rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(m_pose_b_sample_arr[:,3:].T), '.', markersize=3)

        plt.figure('w_M_ω')
        plt.subplot(1,2,1)
        plt.title('w_M_ω trans (m)')
        plt.plot(['tx', 'ty', 'tz'], w_pose_ω_sample_arr[:,:3].T, '.', markersize=3)
        plt.subplot(1,2,2)
        plt.title('w_M_ω rot (deg)')
        plt.plot(['ox', 'oy', 'oz'], np.rad2deg(w_pose_ω_sample_arr[:,3:].T), '.', markersize=3)

        print('MEANS:')
        m_p_mb_mean = np.mean(m_pose_b_sample_arr[:,:3], axis=0)
        m_o_b_mean = np.mean(m_pose_b_sample_arr[:,3:], axis=0)
        m_quat_b_mean = pin.Quaternion(pin.exp(m_o_b_mean)).coeffs()

        w_p_wω_mean = np.mean(w_pose_ω_sample_arr[:,:3], axis=0)
        w_o_ω_mean = np.mean(w_pose_ω_sample_arr[:,3:], axis=0)
        w_quat_ω_mean = pin.Quaternion(pin.exp(w_o_ω_mean)).coeffs()

        offset_mean = np.mean(offset_sample_arr, axis=0)
        friction_mean = np.mean(friction_sample_arr, axis=0)
        alpha_mean = np.mean(alpha_sample_arr, axis=0)

        print('m_p_mb_mean', m_p_mb_mean)
        print('m_quat_b_mean', m_quat_b_mean)
        print('w_p_wω_mean', w_p_wω_mean)
        print('w_quat_ω_mean', w_quat_ω_mean)
        print('offset_mean', offset_mean)
        print('friction_mean', friction_mean)
        print('alpha_mean', alpha_mean)

    def save_calib(self, r, name='calib'):
        alpha = r.x[:12]
        friction = r.x[12:24]
        offset = r.x[24:36]
        nu_b = r.x[-12:-6]
        nu_ω = r.x[-6:]

        m_M_b = self.m_M_b_init*pin.exp(nu_b)
        w_M_ω = self.w_M_ω_init*pin.exp(nu_ω)
        m_p_mb = m_M_b.translation
        m_q_b = pin.Quaternion(m_M_b.rotation).coeffs()
        w_p_wω = w_M_ω.translation
        w_q_ω = pin.Quaternion(w_M_ω.rotation).coeffs()

        d = {
            'alpha': alpha.tolist(), 
            'friction': friction.tolist(), 
            'offset': offset.tolist(), 
            'm_p_mb': m_p_mb.tolist(), 
            'm_q_b': m_q_b.tolist(), 
            'w_p_wω': w_p_wω.tolist(), 
            'w_q_ω': w_q_ω.tolist(), 
        }

        with open(name+'.json', 'w') as outfile:
            json.dump(d, outfile)
            print('Saved', name+'.json')
