from functools import partial
import numpy as np
import pinocchio as pin

def screw(v):
    return np.array([0,   -v[2],  v[1],
                     v[1],   0,  -v[0],
                    -v[1], v[0],  0    ]).reshape((3,3))


class ImuLegKF:

    def __init__(self, robot, dt, contact_ids, x_init, std_prior, 
                 std_foot, std_foot_vel, std_acc, std_wb, std_qa, std_dqa, std_kin, std_hfoot):
        self.robot = robot
        # stores a look up table for contact state position
        print(contact_ids)
        self.cids_idx = {cid: 6+3*i for i, cid in enumerate(contact_ids)}
        print(self.cids_idx)
        # state structure: [base position, base velocity, [feet position] ] 
        # all quantities expressed in universe frame
        # [o_p_ob, o_v_o1, o_p_ol1, o_p_ol1, o_p_ol1, o_p_ol3]
        self.x = x_init
        # helpers to retrieve parts of the state variable

        # state cov
        self.P = np.diag(std_prior)**2

        # discretization period
        self.dt = dt

        # discretized noise std
        self.std_acc = std_acc           # noise on IMU acceleration measurement (if isotropic, does not matter if in body or inertial frame)

        # static jacs and covs
        self.Qacc = std_acc**2 * np.eye(3)
        self.Qwb = std_wb**2 * np.eye(3)
        self.Qqa = std_qa**2 * np.eye(self.robot.model.nv - 6)
        self.Qdqa = std_dqa**2 * np.eye(self.robot.model.nv - 6)
        self.Qfoot = std_foot**2 * np.eye(3)
        self.Qkin = std_kin**2 * np.eye(3)
        self.Qhfoot = std_hfoot**2
        
        # propagation
        self.Fk = self.propagation_jac()
        self.Qk = self.propagation_cov()

        self.H_vel = self.vel_meas_H()
        # self.R_vel = self.std_foot_vel**2 * np.eye(3)
        
        self.H_relp_dic = {cid: self.relp_meas_H(cid_idx) for cid, cid_idx in self.cids_idx.items()}
        self.R_relp = std_foot_vel**2 * np.eye(3)
    
    def get_state(self):
        return self.x

    def propagate(self, o_a_ob, feet_in_contact_ids):
        """
        o_a_ob: acceleration of the base wrt world frame in world frame (oRb*b_am + o_g)
        (IMU proper acceleration rotated in world frame)
        feet_in_contact_ids: options
             - size 4 boolean array
             - 
        """
        # state propagation
        self.x[0:3] += self.x[3:6] * self.dt + 0.5*o_a_ob*self.dt**2
        self.x[3:6] += o_a_ob*self.dt
        
        # cov propagation
        # TODO: adjust Qk depending on feet in contact?
        self.P = self.Fk @ self.P @ self.Fk.T + self.Qk

    def correct(self, qa, dqa, oRb, b_wb, feet_in_contact_ids):
        # just to be sure we do not have base placement/velocity in q and dq
        q = np.concatenate([np.array(6*[0]+[1]), qa])
        dq = np.concatenate([np.zeros(6), dqa])
        # update the robot state, freeflyer at the universe not moving
        self.robot.forwardKinematics(q, dq)
        #################
        # Geometry update
        #################
        # for each foot (contact or not) update position using kinematics
        # Adapt cov if in conact or not
        for cid in self.cids_idx:
            o_p_bl = oRb @ self.robot.framePlacement(q, cid, update_kinematics=False).translation
            R_relp = self.relp_cov(q, oRb, cid)
            self.kalman_update(o_p_bl, self.H_relp_dic[cid], R_relp)

        ################################
        # differential kinematics update
        ################################
        # For feet in contact only, use the zero velocity assumption
        for cid in feet_in_contact_ids:
            # b_T_l = self.robot.framePlacement(q, cid, update_kinematics=False)
            # b_p_bl = b_T_l.translation
            # bRl = b_T_l.rotation

            # l_v_bl = self.robot.frameVelocity(q, dq, cid, update_kinematics=False).linear
            # # measurement: velocity in world frame
            # b_v_ob = - bRl @ l_v_bl + np.cross(b_p_bl, b_wb)
            # o_v_ob = oRb @ b_v_ob
            o_v_ob = base_vel_from_stable_contact(self.robot, q, dq, b_wb, oRb, cid)

            R_vel = self.vel_meas_cov(q, b_wb, oRb, cid)

            self.kalman_update(o_v_ob, self.H_vel, R_vel)

        # ####################################
        # # foot in contact zero height update
        # ####################################
        # for cid in feet_in_contact_ids:
        #     # zero height update
        #     hfoot = 0.0
        #     H = np.zeros(6*3)
        #     H[self.cids_idx[cid]+2] = 1
        #     self.kalman_update(hfoot, H, self.Qhfoot)

    def kalman_update(self, y, H, R):
        # general unoptimized kalman update
        # innovation z = y - h(x)
        # Innov cov Z = HPH’ + R ; avec H = dh/dx = - dz/dx
        # Kalman gain K = PH’ / Z
        # state error dx = K*z
        # State update x <-- x (+) dx
        # Cov update P <-- P - KZP

        z = y - H @ self.x
        Z = H @ self.P @ H.T + R
        if isinstance(Z, np.ndarray):
            K = self.P @ H.T @ np.linalg.inv(Z)
            dx = K @ z
        else:  # for scalar measurements
            K = self.P @ H.T / Z
            dx = K * z
            # reshape to avoid confusion between inner and outer product when multiplying 2 1d arrays
            K = K.reshape((18,1))
            H = H.reshape((1,18))
        self.x = self.x + dx
        self.P = self.P - K @ H @ self.P
        
        return z, dx

    def propagation_jac(self):
        F = np.eye(self.x.shape[0])
        F[0:3,3:6] = self.dt*np.eye(3)
        return F

    def propagation_cov(self):
        Q = np.zeros((self.x.shape[0], self.x.shape[0]))
        Q[0:3,0:3] = self.dt**2 * self.Qacc / 4
        Q[0:3,3:6] = self.dt    * self.Qacc / 2
        Q[3:6,0:3] = self.dt    * self.Qacc / 2
        Q[3:6,3:6] = self.Qacc
        for i_fi in self.cids_idx.values():
            Q[i_fi:i_fi+3,i_fi:i_fi+3] = self.Qfoot
        return Q

    @staticmethod
    def vel_meas_H():
        H = np.zeros((3, 6*3))
        H[0:3,3:6] = np.eye(3)
        return H
    
    def vel_meas_cov(self, q, wb, oRb, cid):
        wbx = screw(wb)
        bTl = self.robot.framePlacement(q, cid, update_kinematics=False)
        b_Jl = bTl.rotation @ self.robot.computeFrameJacobian(q, cid)[:3,6:]
        o_Jl = oRb @ b_Jl 
        b_p_bl_x = screw(bTl.translation)
        return - o_Jl @ self.Qdqa @ o_Jl.T - b_p_bl_x @ self.Qwb @ b_p_bl_x + wbx @ b_Jl @ self.Qqa @ b_Jl.T @ wbx
 
    def relp_cov(self, q, oRb, cid):
        bTl = self.robot.framePlacement(q, cid, update_kinematics=False)
        o_Jl = oRb @ bTl.rotation @ self.robot.computeFrameJacobian(q, cid)[:3,6:]
        return o_Jl @ self.Qdqa @ o_Jl.T + self.Qkin
    
    @staticmethod
    def relp_meas_H(cid_idx):
        H = np.zeros((3,6*3))
        H[:3,:3] = - np.eye(3)
        H[:3,cid_idx:cid_idx+3] = np.eye(3) 
        return H
    


class ImuLegCF:

    def __init__(self, robot, dt, contact_ids, x_init):
        self.robot = robot
        # [o_p_ob, o_v_o1]
        self.x = x_init

        # discretization period
        self.dt = dt

        self.v_imu_int = x_init[3:6]
        self.v_imu_hp =  x_init[3:6]
        self.v_kin_lp =  x_init[3:6]

        self.alpha = 0.99

    def get_state(self):
        return self.x

    def update_state(self, o_acc, qa, dqa, b_wb, oRb, feet_in_contact_ids):
        if len(feet_in_contact_ids) == 0:
            # no feet in contact, nothin to do
            return
            
        # just to be sure we do not have base placement/velocity in q and dq
        q = np.concatenate([np.array(6*[0]+[1]), qa])
        dq = np.concatenate([np.zeros(6), dqa])
        # update the robot state, freeflyer at the universe not moving
        self.robot.forwardKinematics(q, dq)

        # integrate position
        self.x[:3] += self.x[3:6]*self.dt + 0.5*o_acc*self.dt**2

        # get integrated velocity from IMU acceleration
        v_imu_int_prev  = self.v_imu_int.copy()
        self.v_imu_int += o_acc*self.dt 
        self.v_imu_hp = self.alpha*(self.v_imu_int - v_imu_int_prev + self.v_imu_hp)

        # get current velocity mean from contacts
        o_vb_kin_mean = np.zeros(3)
        for cid in feet_in_contact_ids:
            o_vb_kin_mean += base_vel_from_stable_contact(self.robot, q, dq, b_wb, oRb, cid)
        # print(o_vb_kin_mean)
        # print(len(feet_in_contact_ids))
        o_vb_kin_mean /= len(feet_in_contact_ids)
        self.v_kin_lp = self.alpha*self.v_kin_lp + (1 - self.alpha)*o_vb_kin_mean

        self.x[3:6] = self.v_kin_lp + self.v_imu_hp


def base_vel_from_stable_contact(robot, q, dq, b_wb, oRb, cid):
    """
    Assumes forwardKinematics has been called on the robot object with current q dq
    """
    b_T_l = robot.framePlacement(q, cid, update_kinematics=False)
    b_p_bl = b_T_l.translation
    bRl = b_T_l.rotation

    l_v_bl = robot.frameVelocity(q, dq, cid, update_kinematics=False).linear
    # measurement: velocity in world frame
    b_v_ob = - bRl @ l_v_bl + np.cross(b_p_bl, b_wb)
    return oRb @ b_v_ob