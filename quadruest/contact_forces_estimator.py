import numpy as np
from numpy.linalg import pinv
from scipy import stats
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()


class ContactForcesEstimator:
    def __init__(self, robot, contact_frame_id_lst, dt): 
        self.robot = robot
        self.nv = self.robot.model.nv 
        self.contact_frame_id_lst = contact_frame_id_lst
        self.dt = dt
        self.b_omg_ob_prev = np.zeros(3)
        self.dqa_prev = np.zeros(12)

    def compute_contact_forces(self, qa, dqa, o_R_b, b_v_ob, b_omg_ob, o_a_ob, tauj, world_frame=True):
        """ 
        Compute the 3D contact/reaction forces at the solo feet using inverse dynamics.
        Forces are expressed in body frame.
        """
        b_v_ob = np.zeros(3)
        o_quat_b = pin.Quaternion(o_R_b).coeffs()
        q  = np.concatenate([np.array([0,0,0]), o_quat_b, qa])  # robot anywhere with orientation estimated from IMU
        vq = np.concatenate([b_v_ob, b_omg_ob, dqa])                 # generalized velocity in base frame as pin requires
        # vq = np.hstack([np.zeros(3), b_omg_ob, dqa])           # generalized velocity in base frame as pin requires
        b_spa = o_R_b.T @ o_a_ob - np.cross(b_omg_ob, b_v_ob) # spatial acceleration linear part, neglecting the rest
        ddqa = (dqa - self.dqa_prev)/self.dt   # using ddqa numdiff only seems to yield the best results
        # ddqa = np.zeros(12)
        # b_domgdt_ob = (b_omg_ob - self.b_omg_ob_prev)/self.dt
        b_domgdt_ob = np.zeros(3)
        dvq = np.concatenate([b_spa, b_domgdt_ob, ddqa])

        taucf = pin.rnea(self.robot.model, self.robot.data, q, vq, dvq) # sum of joint torque + contact torques corresponding to the actuated equations
        taucf[6:] = taucf[6:] - tauj

        self.dqa_prev = dqa
        self.b_omg_ob_prev = b_omg_ob

        return taucf_to_oforces(self.robot, q, self.nv, o_R_b, taucf, self.contact_frame_id_lst, world_frame=world_frame)


    def compute_contact_forces2(self, qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_oi, o_a_oi, tauj, world_frame=True):
        """ 
        Compute the 3D contact/reaction forces at the solo feet using inverse dynamics.
        Forces are expressed in body frame.
        """
        zero3 = np.array([0,0,0])
        o_quat_i = pin.Quaternion(o_R_i).coeffs()
        q  = np.concatenate([zero3, o_quat_i, qa])   # robot anywhere with orientation estimated from IMU
        vq = np.concatenate([zero3, i_omg_oi, dqa])             # generalized velocity in base frame as pin requires
        i_a_oi = o_R_i.T @ o_a_oi # - np.cross(b_omg_ob, b_v_ob) # since the i_v_oi is set to zero arbitrarily
        dvq = np.concatenate([i_a_oi, i_domg_oi, ddqa])

        taucf = pin.rnea(self.robot.model, self.robot.data, q, vq, dvq) # sum of joint torque + contact torques corresponding to the actuated equations
        taucf[6:] = taucf[6:] - tauj

        self.dqa_prev = dqa
        self.i_omg_oi_prev = i_omg_oi

        return taucf_to_oforces(self.robot, q, self.nv, o_R_i, taucf, self.contact_frame_id_lst, world_frame=world_frame)



class ContactForceEstimatorGeneralizedMomentum:
    def __init__(self, robot, contact_frame_id_lst, dt):
        self.robot = robot
        self.rm = robot.model
        self.rd = robot.data
        self.nv = self.robot.model.nv 
        self.contact_frame_id_lst = contact_frame_id_lst
        self.dt = dt

        # parameters of the filter:
        self.Ki = 70
        self.r = np.zeros(self.nv)  # first order
        self.p0 = np.zeros(self.nv)  # not necessary -> to compute
        self.int = np.zeros(self.nv)

    def compute_contact_forces(self, qa, dqa, o_R_b, b_v_ob, b_omg_ob, _, tauj, world_frame=True):
        o_quat_b = pin.Quaternion(o_R_b).coeffs()
        q =  np.concatenate([np.array([0,0,0]), o_quat_b, qa])  # robot anywhere with orientation estimated from IMU
        vq = np.concatenate([b_v_ob, b_omg_ob, dqa])
        # vq = np.hstack([np.zeros(3), b_omg_ob, dqa])
        C = pin.computeCoriolisMatrix(self.rm, self.rd, q, vq)
        g = pin.computeGeneralizedGravity(self.rm, self.rd, q)
        M = pin.crba(self.rm, self.rd, q)
        p = M @ vq
        tauj = np.concatenate([np.zeros(6), tauj])
        self.int = self.int + (tauj + C.T @ vq - g + self.r)*self.dt
        self.r = self.Ki * (p - self.int - self.p0)

        return taucf_to_oforces(self.robot, q, self.nv, o_R_b, self.r, self.contact_frame_id_lst)


def taucf_to_oforces(robot, q, nv, o_R_b, taucf, contact_frame_id_lst, world_frame=True):
    Jlinvel = compute_joint_jac(robot, q, contact_frame_id_lst, world_frame=world_frame)

    # 3 options, same outcomes IF the indices are in the right order for the 3rd:
    # 1) solve the whole system:
    o_forces = np.linalg.pinv(Jlinvel.T) @ taucf

    # 2) remove the "freeflyer" equations:
    # Jlinvel_without_freflyer = Jlinvel[:,6:]
    # taucf_without_freel = taucf[6:]
    # o_forces = np.linalg.pinv(Jlinvel_without_freflyer.T) @ taucf_without_freel

    # 3) Use only leg submatrices
    # Jlinvel_without_freflyer = Jlinvel[:,6:]
    # taucf_without_freel = taucf[6:]
    # o_forces = np.zeros(12)
    # for i in range(4):
    #     o_forces[3*i:3*i+3] = np.linalg.solve(Jlinvel_without_freflyer[3*i:3*i+3,3*i:3*i+3].T, taucf_without_freel[3*i:3*i+3]) 

    # reshape from
    # [f0x, f0y, f0z, f1x, ..., f3z]
    # to
    # [[f0x, f1x, f2x, f3x]
    #        ...y...
    #        ...z...      ]]
    # return o_forces.reshape((4,3)).T
    return o_forces.reshape((len(contact_frame_id_lst),3))


def compute_joint_jac(robot, q, cf_ids, world_frame=True):
    Jlinvel = np.zeros((len(cf_ids)*3, robot.nv))
    for i, frame_id in enumerate(cf_ids):
        if world_frame:
            oTl = robot.framePlacement(q, frame_id, update_kinematics=True)
            Jlinvel[3*i:3*(i+1),:] = oTl.rotation @ robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in world coord
        else: 
            Jlinvel[3*i:3*(i+1),:] = robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in local coord
    return Jlinvel


class ContactDetection:

    def __init__(self, fmin, vmax, std_fz, std_vz, proba_thresh):
        self.nfz = stats.norm(fmin, std_fz)
        self.nvz = stats.norm(vmax, std_vz)
        self.proba_thresh = proba_thresh

    def contact_proba(self, fz, vz):
        """
        Try to fuse instantaneous feet force and velocity in world frame. 
        Floor assumed to be flat and orthogonal to Z.
        """
        Pfz = self.nfz.cdf(fz)
        Pvz = 1 - self.nvz.cdf(vz)

        # return (Pfz * Pvz) > self.proba_thresh
        return Pfz, Pvz, (Pfz * Pvz)
    
    def contact_detection(self, fz, vz):
        return self.contact_detection(fz, vz) > self.proba_thresh

    