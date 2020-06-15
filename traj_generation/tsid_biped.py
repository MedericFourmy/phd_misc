"""
Modified from tsid/exercizes/tsid_pided
"""

import eigenpy
eigenpy.switchToNumpyArray()
import pinocchio as se3
from pinocchio import libpinocchio_pywrap as pin 
import tsid
import numpy as np
from numpy.linalg import norm
import os
import gepetto.corbaserver
import time
import subprocess


class TsidBiped:
    ''' Standard TSID formulation for a biped robot standing on its rectangular feet.
        - Center of mass task
        - Postural task
        - 6d rigid contact constraint for both feet
        - Regularization task for contact forces
    '''
    
    def __init__(self, conf, viewer=True):
        self.conf = conf
        print('Robot files:')
        print(conf.urdf)
        print(conf.srdf)
        vector = se3.StdVec_StdString()
        vector.extend(item for item in conf.path)
        self.robot = tsid.RobotWrapper(conf.urdf, vector, se3.JointModelFreeFlyer(), False)
        robot = self.robot
        self.model = robot.model()

        pin.loadReferenceConfigurations(self.model, conf.srdf, False)
        self.q0 = q = self.model.referenceConfigurations["half_sitting"]
        v = np.zeros(robot.nv)
        
        assert self.model.existFrame(conf.rf_frame_name)
        assert self.model.existFrame(conf.lf_frame_name)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        # contact_Point = np.ones((3,4)) * conf.lz
        contact_Point = -np.ones((3,4)) * conf.lz
        contact_Point[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
        contact_Point[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]
        
        contactRF = tsid.Contact6d("contact_rfoot", robot, conf.rf_frame_name, contact_Point, 
                                  conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
        contactRF.setKp(conf.kp_contact * np.ones(6).T)
        contactRF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6).T)
        self.RF = robot.model().getJointId(conf.rf_frame_name)
        H_rf_ref = robot.position(data, self.RF)
        
        # modify initial robot configuration so that foot is on the ground (z=0)
        q[2] -= H_rf_ref.translation[2] - conf.lz
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        H_rf_ref = robot.position(data, self.RF)
        contactRF.setReference(H_rf_ref)
        formulation.addRigidContact(contactRF, conf.w_forceRef)
        
        contactLF =tsid.Contact6d("contact_lfoot", robot, conf.lf_frame_name, contact_Point, 
                                  conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
        contactLF.setKp(conf.kp_contact * np.ones(6).T)
        contactLF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6).T)
        self.LF = robot.model().getJointId(conf.lf_frame_name)
        H_lf_ref = robot.position(data, self.LF)
        contactLF.setReference(H_lf_ref)
        formulation.addRigidContact(contactLF, conf.w_forceRef)
        
        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        formulation.addMotionTask(comTask, conf.w_com, 0, 0.0)
        
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        # self.leftFootTask = tsid.TaskSE3Equality("task-left-foot", self.robot, self.conf.lf_frame_name)
        # self.leftFootTask.setKp(self.conf.kp_foot * np.ones(6))
        # self.leftFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6))
        self.trajLF = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)
        # formulation.addMotionTask(self.leftFootTask, self.conf.w_foot, 1, 0.0)
        
        # self.rightFootTask = tsid.TaskSE3Equality("task-right-foot", self.robot, self.conf.rf_frame_name)
        # self.rightFootTask.setKp(self.conf.kp_foot * np.ones(6))
        # self.rightFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6).T)
        self.trajRF = tsid.TrajectorySE3Constant("traj-right-foot", H_rf_ref)
        # formulation.addMotionTask(self.rightFootTask, self.conf.w_foot, 1, 0.0)
        
        # self.tau_max = conf.tau_max_scaling*robot.model().effortLimit[-robot.na:]
        # self.tau_min = -self.tau_max
        # actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        # actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        # if(conf.w_torque_bounds>0.0):
        #     formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
            
        # jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
        # self.v_max = conf.v_max_scaling * robot.model().velocityLimit[-robot.na:]
        # self.v_min = -self.v_max
        # jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        # if(conf.w_joint_bounds>0.0):
        #     formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)
        
        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()
        
        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        postureTask.setReference(self.trajPosture.computeNext())
        
        self.sampleLF  = self.trajLF.computeNext()
        self.sample_LF_pos = self.sampleLF.pos()
        self.sample_LF_vel = self.sampleLF.vel()
        self.sample_LF_acc = self.sampleLF.acc()
        
        self.sampleRF  = self.trajRF.computeNext()
        self.sample_RF_pos = self.sampleRF.pos()
        self.sample_RF_vel = self.sampleRF.vel()
        self.sample_RF_acc = self.sampleRF.acc()
        
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.comTask = comTask
        self.postureTask = postureTask
        self.contactRF = contactRF
        self.contactLF = contactLF
        # self.actuationBoundsTask = actuationBoundsTask
        # self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.q = q
        self.v = v
        
        self.contact_LF_active = True
        self.contact_RF_active = True
        
        # for gepetto viewer
        if(viewer):
            self.robot_display = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
            l = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(l[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(loadModel=True)
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(q)
            
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform('python-pinocchio', conf.CAMERA_TRANSFORM)
            self.gui.addFloor('world/floor')
            self.gui.setLightingMode('world/floor', 'OFF')

            self.gui.addSphere('world/com', conf.SPHERE_RADIUS, conf.COM_SPHERE_COLOR)
            self.gui.addSphere('world/com_ref', conf.REF_SPHERE_RADIUS, conf.COM_REF_SPHERE_COLOR)
            self.gui.addSphere('world/rf', conf.SPHERE_RADIUS, conf.RF_SPHERE_COLOR)
            self.gui.addSphere('world/rf_ref', conf.REF_SPHERE_RADIUS, conf.RF_REF_SPHERE_COLOR)
            self.gui.addSphere('world/lf', conf.SPHERE_RADIUS, conf.LF_SPHERE_COLOR)
            self.gui.addSphere('world/lf_ref', conf.REF_SPHERE_RADIUS, conf.LF_REF_SPHERE_COLOR)
        
    # def integrate_dv(self, q, v, dv, dt):
    #     v_mean = v + 0.5*dt*dv
    #     v += dt*dv
    #     q = se3.integrate(self.model, q, dt*v_mean)
    #     return q,v

    def integrate_dv(self, q, v, dv, dt):
        q = pin.integrate(self.model, q, dt*v)
        v += dt*dv
        return q, v

    def print_solve_check(self, sol, t, v, dv):
        print("Time %.3f"%(t))
        if self.formulation.checkContact(self.contactRF.name, sol):
            f = self.formulation.getContactForce(self.contactRF.name, sol)
            print ("\tnormal force %s: %.1f"%(self.contactRF.name.ljust(20,'.'), self.contactRF.getNormalForce(f)))

        if self.formulation.checkContact(self.contactLF.name, sol):
            f = self.formulation.getContactForce(self.contactLF.name, sol)
            print ("\tnormal force %s: %.1f"%(self.contactLF.name.ljust(20,'.'), self.contactLF.getNormalForce(f)))
 
        print ("\ttracking err %s: %.3f"%(self.comTask.name.ljust(20,'.'), norm(self.comTask.position_error, 2)))
        print ("\t||v||: %.3f\t ||dv||: %.3f"%(norm(v, 2), norm(dv)))
    
    def update_display(self, q, t):
        self.robot_display.display(q)
        x_com = self.robot.com(self.formulation.data())
        x_com_ref = self.trajCom.getSample(t).pos()
        H_lf = self.robot.position(self.formulation.data(), self.LF)
        H_rf = self.robot.position(self.formulation.data(), self.RF)
        x_lf_ref = self.trajLF.getSample(t).pos()[:3]
        x_rf_ref = self.trajRF.getSample(t).pos()[:3]
        self.gui.applyConfiguration('world/com', x_com.tolist()+[0,0,0,1.])
        self.gui.applyConfiguration('world/com_ref', x_com_ref.tolist()+[0,0,0,1.])
        self.gui.applyConfiguration('world/rf', pin.SE3ToXYZQUATtuple(H_rf))
        self.gui.applyConfiguration('world/lf', pin.SE3ToXYZQUATtuple(H_lf))
        self.gui.applyConfiguration('world/rf_ref', x_rf_ref.tolist()+[0,0,0,1.])
        self.gui.applyConfiguration('world/lf_ref', x_lf_ref.tolist()+[0,0,0,1.])

    def get_placement_LF(self):
        return self.robot.position(self.formulation.data(), self.LF)
        
    def get_placement_RF(self):
        return self.robot.position(self.formulation.data(), self.RF)
        
    def set_com_ref(self, pos, vel, acc):
        self.sample_com.pos(pos)
        self.sample_com.vel(vel)
        self.sample_com.acc(acc)
        self.comTask.setReference(self.sample_com)
        
    def set_RF_3d_ref(self, pos, vel, acc):
        self.sample_RF_pos[:3,0] = pos
        self.sample_RF_vel[:3,0] = vel
        self.sample_RF_acc[:3,0] = acc
        self.sampleRF.pos(self.sample_RF_pos)
        self.sampleRF.vel(self.sample_RF_vel)
        self.sampleRF.acc(self.sample_RF_acc)        
        self.rightFootTask.setReference(self.sampleRF)
        
    def set_LF_3d_ref(self, pos, vel, acc):
        self.sample_LF_pos[:3,0] = pos
        self.sample_LF_vel[:3,0] = vel
        self.sample_LF_acc[:3,0] = acc
        self.sampleLF.pos(self.sample_LF_pos)
        self.sampleLF.vel(self.sample_LF_vel)
        self.sampleLF.acc(self.sample_LF_acc)        
        self.leftFootTask.setReference(self.sampleLF)
        
    def get_LF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H  = self.robot.position(data, self.LF)
        v  = self.robot.velocity(data, self.LF)
        a = self.leftFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]
        
    def get_RF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H  = self.robot.position(data, self.RF)
        v  = self.robot.velocity(data, self.RF)
        a = self.rightFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]
        
    def remove_contact_RF(self, transition_time=0.0):
        H_rf_ref = self.robot.position(self.formulation.data(), self.RF)
        self.trajRF.setReference(H_rf_ref)
        self.rightFootTask.setReference(self.trajRF.computeNext())
    
        self.formulation.removeRigidContact(self.contactRF.name, transition_time)
        self.contact_RF_active = False
        
    def remove_contact_LF(self, transition_time=0.0):        
        H_lf_ref = self.robot.position(self.formulation.data(), self.LF)
        self.trajLF.setReference(H_lf_ref)
        self.leftFootTask.setReference(self.trajLF.computeNext())
        
        self.formulation.removeRigidContact(self.contactLF.name, transition_time)
        self.contact_LF_active = False
        
    def add_contact_RF(self, transition_time=0.0):       
        H_rf_ref = self.robot.position(self.formulation.data(), self.RF)
        self.contactRF.setReference(H_rf_ref)
        self.formulation.addRigidContact(self.contactRF, self.conf.w_forceRef)
        
        self.contact_RF_active = True
        
    def add_contact_LF(self, transition_time=0.0):        
        H_lf_ref = self.robot.position(self.formulation.data(), self.LF)
        self.contactLF.setReference(H_lf_ref)
        self.formulation.addRigidContact(self.contactLF, self.conf.w_forceRef)
        
        self.contact_LF_active = True