"""
Modified from tsid/exercizes/tsid_pided
"""

import eigenpy
eigenpy.switchToNumpyArray()
# import pinocchio as pin
# from pinocchio import libpinocchio_pywrap as pin 
import pinocchio as pin
import tsid
import numpy as np
from numpy.linalg import norm
import os
import gepetto.corbaserver
import time
import subprocess


class TsidWrapper:
    """ Standard TSID invdyn for a biped robot standing on its rectangular feet.
        - Center of mass task
        - Postural task
        - 6d rigid contact constraint for both feet
        - Regularization task for contact forces
    """
    
    def __init__(self, conf, viewer=True):
        self.conf = conf
        self.contact_frame_names = conf.contact_frame_names
        print('Robot files:')
        print(conf.urdf)
        print(conf.srdf)
        vector = pin.StdVec_StdString()
        vector.extend(item for item in conf.path)
        self.robot = tsid.RobotWrapper(conf.urdf, vector, pin.JointModelFreeFlyer(), False)
        robot = self.robot
        self.model = robot.model()

        pin.loadReferenceConfigurations(self.model, conf.srdf, False)
        self.q0 = q = self.model.referenceConfigurations[conf.reference_config_q_name]
        v = np.zeros(robot.nv)

        assert(all([self.model.existFrame(frame_name) for frame_name in self.contact_frame_names]))
        
        invdyn = tsid.InverseDynamicsFormulationAccForce('tsid', robot, False)
        invdyn.computeProblemData(0.0, q, v)
        data = invdyn.data()

        robot.computeAllTerms(data, q, v)

        self.contact_frame_ids = [self.model.getFrameId(frame_name) for frame_name in self.contact_frame_names]
        self.nc = len(self.contact_frame_ids)  # contact number

        self.contacts = self.nc*[None]
        self.tasks_tracking_foot = self.nc*[None]
        self.traj_feet = self.nc*[None]
        self.sample_feet = self.nc*[None]
        mask = np.ones(6)
        if not conf.contact6d:
            mask[3:] = 0
        for i_foot, (frame_name, fid) in enumerate(zip(self.contact_frame_names, self.contact_frame_ids)):
            Hf_ref = robot.framePosition(data, fid)
            
            if conf.contact6d:
                # DEFINE FOOT CONTACT POINTS LOCATION WRT LOCAL FOOT FRAMES
                # contact_points = np.ones((3,4)) * conf.lz
                contact_points = -np.ones((3,4)) * conf.lz
                contact_points[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
                contact_points[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]
                
                # RIGID CONTACT RIGHT FOOT
                self.contacts[i_foot] = tsid.Contact6d('contact_{}'.format(frame_name), robot, frame_name, contact_points, 
                                        conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
                self.contacts[i_foot].setKp(conf.kp_contact * np.ones(6).T)
                self.contacts[i_foot].setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6).T)
                self.contacts[i_foot].setReference(Hf_ref)
                invdyn.addRigidContact(self.contacts[i_foot], conf.w_forceRef)
                
            else:
                # RIGID POINT CONTACTS
                self.contacts[i_foot] = tsid.ContactPoint('contact_{}'.format(frame_name), robot, frame_name, conf.contactNormal, conf.mu, conf.fMin, conf.fMax)
                self.contacts[i_foot].setKp(conf.kp_contact * np.ones(3))
                self.contacts[i_foot].setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(3))
                self.contacts[i_foot].setReference(Hf_ref)
                self.contacts[i_foot].useLocalFrame(False)
                invdyn.addRigidContact(self.contacts[i_foot], conf.w_forceRef)

            # FEET TRACKING TASKS
            self.tasks_tracking_foot[i_foot] = tsid.TaskSE3Equality(
                'task-foot{}'.format(i_foot), self.robot, self.contact_frame_names[i_foot])
            self.tasks_tracking_foot[i_foot].setKp(conf.kp_foot * mask)
            self.tasks_tracking_foot[i_foot].setKd(2.0 * np.sqrt(conf.kp_foot) * mask)
            self.tasks_tracking_foot[i_foot].setMask(mask)
            self.tasks_tracking_foot[i_foot].useLocalFrame(False)
            invdyn.addMotionTask(self.tasks_tracking_foot[i_foot], conf.w_foot, conf.priority_foot, 0.0)

            self.traj_feet[i_foot] = tsid.TrajectorySE3Constant('traj-foot{}'.format(i_foot), Hf_ref)
            self.sample_feet[i_foot] = self.traj_feet[i_foot].computeNext()

        # COM TASK
        comTask = tsid.TaskComEquality('task-com', robot)
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        invdyn.addMotionTask(comTask, conf.w_com, conf.priority_com, 1.0)

        # POSTURE TASK
        postureTask = tsid.TaskJointPosture('task-posture', robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        invdyn.addMotionTask(postureTask, conf.w_posture, conf.priority_posture, 0.0)

        # # WAIST Task
        # waistTask = tsid.TaskSE3Equality("keepWaist", robot, 'root_joint') # waist -> root_joint
        # waistTask.setKp(conf.kp_waist * np.ones(6)) # Proportional gain defined before = 500
        # waistTask.setKd(2.0 * np.sqrt(conf.kp_waist) * np.ones(6)) # Derivative gain = 2 * sqrt(500), critical damping
        # # Add a Mask to the task which will select the vector dimensions on which the task will act.
        # # In this case the waist configuration is a vector 6d (position and orientation -> SE3)
        # # Here we set a mask = [0 0 0 1 1 1] so the task on the waist will act on the orientation of the robot
        # mask = np.ones(6)
        # # mask[:3] = 0.
        # waistTask.setMask(mask)
        # invdyn.addMotionTask(waistTask, conf.w_waist, conf.priority_waist, 0.0)
        
        # TORQUE BOUND TASK
        # self.tau_max = conf.tau_max_scaling*robot.model().effortLimit[-robot.na:]
        # self.tau_min = -self.tau_max
        # actuationBoundsTask = tsid.TaskActuationBounds('task-actuation-bounds', robot)
        # actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        # if(conf.w_torque_bounds>0.0):
        #     invdyn.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, conf.priority_torque_bounds, 0.0)
            
        # JOINT BOUND TASK
        # jointBoundsTask = tsid.TaskJointBounds('task-joint-bounds', robot, conf.dt)
        # self.v_max = conf.v_max_scaling * robot.model().velocityLimit[-robot.na:]
        # self.v_min = -self.v_max
        # jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        # if(conf.w_joint_bounds>0.0):
        #     invdyn.addMotionTask(jointBoundsTask, conf.w_joint_bounds, conf.priority_joint_bounds, 0.0)
        
        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant('traj_com', com_ref)
        self.sample_com = self.trajCom.computeNext()
        
        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant('traj_joint', q_ref)
        postureTask.setReference(self.trajPosture.computeNext())
        
        self.sample_feet = self.nc*[None]
        self.sample_feet_pos = self.nc*[None]
        self.sample_feet_vel = self.nc*[None]
        self.sample_feet_acc = self.nc*[None]
        for i_foot, traj in enumerate(self.traj_feet):
            self.sample_feet[i_foot] = traj.computeNext()
            self.sample_feet_pos[i_foot] = self.sample_feet[i_foot].pos()
            self.sample_feet_vel[i_foot] = self.sample_feet[i_foot].vel()
            self.sample_feet_acc[i_foot] = self.sample_feet[i_foot].acc()
        
        self.solver = tsid.SolverHQuadProgFast('qp solver')
        self.solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)
        
        self.comTask = comTask
        self.postureTask = postureTask
        # self.actuationBoundsTask = actuationBoundsTask
        # self.jointBoundsTask = jointBoundsTask
        self.invdyn = invdyn
        self.q = q
        self.v = v
        
        self.active_contacts = self.nc*[True]
        
        # for gepetto viewer
        if(viewer):
            self.robot_display = pin.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], pin.JointModelFreeFlyer())
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
            for frame_name in self.contact_frame_names:
                self.gui.addSphere('world/{}'.format(frame_name), conf.SPHERE_RADIUS, conf.F_SPHERE_COLOR)
                self.gui.addSphere('world/{}_ref'.format(frame_name), conf.REF_SPHERE_RADIUS, conf.F_REF_SPHERE_COLOR)

    # def integrate_dv(self, q, v, dv, dt):
    #     v_mean = v + 0.5*dt*dv
    #     v += dt*dv
    #     q = pin.integrate(self.model, q, dt*v_mean)
    #     return q,v

    def integrate_dv(self, q, v, dv, dt):
        q = pin.integrate(self.model, q, dt*v)
        v += dt*dv
        return q, v

    def compute_and_solve(self, t, q, v):
        HQPData = self.invdyn.computeProblemData(t, q, v)
        sol = self.solver.solve(HQPData)
        return sol, HQPData

    def print_solve_check(self, sol, t, v, dv):
        print('Time %.3f'%(t))
        for contact in self.contacts:
            if self.invdyn.checkContact(contact.name, sol):
                f = self.invdyn.getContactForce(contact.name, sol)
                print ('\tnormal force %s: %.1f'%(contact.name.ljust(20,'.'), contact.getNormalForce(f)))

        print ('\ttracking err %s: %.3f'%(self.comTask.name.ljust(20,'.'), norm(self.comTask.position_error, 2)))
        print ('\t||v||: %.3f\t ||dv||: %.3f'%(norm(v, 2), norm(dv)))

    
    def update_display(self, q, t):
        self.robot_display.display(q)
        x_com = self.robot.com(self.invdyn.data())
        x_com_ref = self.comTask.position_ref
        self.gui.applyConfiguration('world/com', x_com.tolist()+[0,0,0,1.])
        self.gui.applyConfiguration('world/com_ref', x_com_ref.tolist()+[0,0,0,1.])
        # for traj, fid, frame_name in zip(self.traj_feet, self.contact_frame_ids, self.contact_frame_names):
        for i_foot, frame_name in enumerate(self.contact_frame_names):
            x_f = self.tasks_tracking_foot[i_foot].position[:3]
            x_f_ref = self.tasks_tracking_foot[i_foot].position_ref[:3]
            self.gui.applyConfiguration('world/{}'.format(frame_name),  x_f.tolist()+[0,0,0,1.])
            self.gui.applyConfiguration('world/{}_ref'.format(frame_name), x_f_ref.tolist()+[0,0,0,1.])

    def get_placement_foot(self, i):
        return self.robot.framePosition(self.invdyn.data(), self.contact_frame_ids[i])
        
    def set_com_ref(self, pos, vel, acc):
        self.sample_com.pos(pos)
        self.sample_com.vel(vel)
        self.sample_com.acc(acc)
        self.comTask.setReference(self.sample_com)
        
    def set_foot_3d_ref(self, pos, vel, acc, i):
        self.sample_feet_pos[i][:3] = pos
        self.sample_feet_vel[i][:3] = vel
        self.sample_feet_acc[i][:3] = acc
        self.sample_feet[i].pos(self.sample_feet_pos[i])
        self.sample_feet[i].vel(self.sample_feet_vel[i])
        self.sample_feet[i].acc(self.sample_feet_acc[i])        
        self.tasks_tracking_foot[i].setReference(self.sample_feet[i])

    def get_3d_pos_vel_acc(self, dv, i):
        data = self.invdyn.data()
        H  = self.robot.framePosition(data, self.contact_frame_ids[i])
        v  = self.robot.velocity(data, self.contact_frame_ids[i])
        a = self.tasks_tracking_foot[i].getAcceleration(dv)
        return H.translation, v.linear, a[:3]
        
    def remove_contact(self, i, transition_time=0.0):
        H_ref = self.robot.framePosition(self.invdyn.data(), self.contact_frame_ids[i])
        self.traj_feet[i].setReference(H_ref)
        self.tasks_tracking_foot[i].setReference(self.traj_feet[i].computeNext())
        self.invdyn.removeRigidContact(self.contacts[i].name, transition_time)
        self.active_contacts[i] = False
             
    def add_contact(self, i, transition_time=0.0):        
        H_ref = self.robot.framePosition(self.invdyn.data(), self.contact_frame_ids[i])
        self.contacts[i].setReference(H_ref)
        self.invdyn.addRigidContact(self.contacts[i], self.conf.w_forceRef)
        self.active_contacts[i] = True