import numpy as np
from numpy.linalg import pinv
from scipy import stats
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()


class ContactForcesEstimator:
    def __init__(self, robot, contact_frame_id_lst): 
        self.robot = robot
        self.nv = self.robot.model.nv 
        self.contact_frame_id_lst = contact_frame_id_lst

    def compute_contact_forces(self, qa, vqa, oRb, b_vb, b_wb, o_acc, tauj):
        """ 
        Compute the 3D contact/reaction forces at the 4 solo feet using inverse dynamics.
        Forces are expressed in body frame.
        """
        o_quat_b = pin.Quaternion(oRb).coeffs()
        q  = np.concatenate([np.array([0,0,0]), o_quat_b, qa])  # robot anywhere with orientation estimated from IMU
        vq = np.concatenate([b_vb, b_wb, vqa])                 # generalized velocity in base frame as pin requires
        # vq = np.hstack([np.zeros(3), b_wb, vqa])           # generalized velocity in base frame as pin requires
        dvq = np.zeros(self.nv)
        dvq[:3] = oRb.T @ o_acc - np.cross(b_wb, b_vb) # spatial acceleration linear part

        taucf = pin.rnea(self.robot.model, self.robot.data, q, vq, dvq) # sum of joint torque + contact torques corresponding to the actuated equations
        taucf[6:] = taucf[6:] - tauj

        return taucf_to_oforces(self.robot, q, self.nv, oRb, taucf, self.contact_frame_id_lst)


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

    def compute_contact_forces(self, qa, vqa, oRb, b_vb, b_wb, o_acc, tauj):
        o_quat_b = pin.Quaternion(oRb).coeffs()
        q =  np.concatenate([np.array([0,0,0]), o_quat_b, qa])  # robot anywhere with orientation estimated from IMU
        vq = np.concatenate([b_vb, b_wb, vqa])                 # generalized velocity in base frame as pin requires
        # vq = np.hstack([np.zeros(3), b_wb, vqa])           # generalized velocity in base frame as pin requires
        C = pin.computeCoriolisMatrix(self.rm, self.rd, q, vq)
        g = pin.computeGeneralizedGravity(self.rm, self.rd, q)
        M = pin.crba(self.rm, self.rd, q)
        p = M @ vq
        tauj = np.concatenate([np.zeros(6), tauj])
        self.int = self.int + (tauj + C.T @ vq - g + self.r)*self.dt
        self.r = self.Ki * (p - self.int - self.p0)

        return taucf_to_oforces(self.robot, q, self.nv, oRb, self.r, self.contact_frame_id_lst)


def taucf_to_oforces(robot, q, nv, oRb, taucf, contact_frame_id_lst):
    Jlinvel = compute_joint_jac(robot, q, contact_frame_id_lst)


    # 3 options, same outcomes IF the indices are in the right order for the 3rd:
    # 1) solve the whole system:
    # o_forces = np.linalg.pinv(Jlinvel.T) @ taucf

    # 2) remove the "freeflyer" equations:
    # Jlinvel_without_freflyer = Jlinvel[:,6:]
    # taucf_without_freel = taucf[6:]
    # o_forces = np.linalg.pinv(Jlinvel_without_freflyer.T) @ taucf_without_freel

    # 3) Use only leg submatrices
    Jlinvel_without_freflyer = Jlinvel[:,6:]
    taucf_without_freel = taucf[6:]
    o_forces = np.zeros(12)
    for i in range(4):
        o_forces[3*i:3*i+3] = np.linalg.solve(Jlinvel_without_freflyer[3*i:3*i+3,3*i:3*i+3].T, taucf_without_freel[3*i:3*i+3]) 

    # [[f0x, f1x, f2x, f3x]
    #        ...y...
    #        ...z...      ]]
    return o_forces.reshape((4,3)).T

def compute_joint_jac(robot, q, cf_ids, world_frame=True):
    Jlinvel = np.zeros((12, robot.nv))
    for i, frame_id in enumerate(cf_ids):
        if world_frame:
            oTl = robot.framePlacement(q, frame_id, update_kinematics=False)
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

    