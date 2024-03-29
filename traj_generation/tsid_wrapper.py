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


        #####################################
        # define contact and associated tasks
        #####################################

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
                self.contacts[i_foot].useLocalFrame(conf.useLocalFrame)
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


        #############
        # Other tasks
        #############

        # COM TASK
        self.comTask = tsid.TaskComEquality('task-com', robot)
        self.comTask.setKp(conf.kp_com * np.ones(3))
        self.comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        invdyn.addMotionTask(self.comTask, conf.w_com, conf.priority_com, 1.0)

        # POSTURE TASK
        self.postureTask = tsid.TaskJointPosture('task-posture', robot)
        self.postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        self.postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        invdyn.addMotionTask(self.postureTask, conf.w_posture, conf.priority_posture, 0.0)

        # TRUNK Task
        self.trunkTask = tsid.TaskSE3Equality("keepTrunk", robot, 'root_joint') 
        self.trunkTask.setKp(conf.kp_trunk * np.ones(6))
        self.trunkTask.setKd(2.0 * np.sqrt(conf.kp_trunk) * np.ones(6))
        # Add a Mask to the task which will select the vector dimensions on which the task will act.
        # In this case the trunk configuration is a vector 6d (position and orientation -> SE3)
        # Here we set a mask = [0 0 0 1 1 1] so the task on the trunk will act on the orientation of the robot
        mask = np.ones(6)
        mask[:3] = 0.
        self.trunkTask.useLocalFrame(False)
        self.trunkTask.setMask(mask)
        invdyn.addMotionTask(self.trunkTask, conf.w_trunk, conf.priority_trunk, 0.0)
        
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
        

        ##################################
        # Init task reference trajectories
        ##################################
        
        self.sample_feet = self.nc*[None]
        self.sample_feet_pos = self.nc*[None]
        self.sample_feet_vel = self.nc*[None]
        self.sample_feet_acc = self.nc*[None]
        for i_foot, traj in enumerate(self.traj_feet):
            self.sample_feet[i_foot] = traj.computeNext()
            self.sample_feet_pos[i_foot] = self.sample_feet[i_foot].value()
            self.sample_feet_vel[i_foot] = self.sample_feet[i_foot].derivative()
            self.sample_feet_acc[i_foot] = self.sample_feet[i_foot].second_derivative()

        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant('traj_com', com_ref)
        self.sample_com = self.trajCom.computeNext()
        
        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant('traj_joint', q_ref)
        self.postureTask.setReference(self.trajPosture.computeNext())
        
        self.trunk_link_id = self.model.getFrameId('base_link')
        R_trunk_init = robot.framePosition(data, self.trunk_link_id).rotation
        self.trunk_ref = self.robot.framePosition(data, self.trunk_link_id)
        self.trajTrunk = tsid.TrajectorySE3Constant("traj_base_link", self.trunk_ref)
        self.sample_trunk = self.trajTrunk.computeNext()
        pos_trunk = np.hstack([np.zeros(3), R_trunk_init.flatten()])
        self.sample_trunk.value()
        self.sample_trunk.derivative(np.zeros(6))
        self.sample_trunk.second_derivative(np.zeros(6))
        self.trunkTask.setReference(self.sample_trunk)



        self.solver = tsid.SolverHQuadProgFast('qp solver')
        self.solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)
        
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

    def integrate_dv_R3SO3(self, q, v, dv, dt):
        b_v = v[0:3]
        b_w = v[3:6]
        b_acc = dv[0:3] + np.cross(b_w, b_v)
        
        p_int = q[:3]
        oRb_int = pin.Quaternion(q[3:7].reshape((4,1))).toRotationMatrix()
        v_int = oRb_int@v[:3]

        p_int = p_int + v_int*dt + 0.5*oRb_int @ b_acc*dt**2
        v_int = v_int + oRb_int @ b_acc*dt
        oRb_int = oRb_int @ pin.exp(b_w*dt)

        q[:3] = p_int
        q[3:7] = pin.Quaternion(oRb_int).coeffs()
        q[7:] += v[6:]*dt
        v += dt*dv
        v[:3] = oRb_int.T@v_int
        return q, v



    def compute_and_solve(self, t, q, v):
        data = self.invdyn.data()
        for i_foot, fid in enumerate(self.contact_frame_ids):
            f_n = data.oMf[fid].rotation[-1,:]  ### normal in local frame
            self.contacts[i_foot].setContactNormal(f_n)
            self.contacts[i_foot].setForceReference(f_n*24.52/4)

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
        self.sample_com.value(pos)
        self.sample_com.derivative(vel)
        self.sample_com.second_derivative(acc)
        self.comTask.setReference(self.sample_com)
    
    def set_trunk_ref(self, R):
        pos = np.hstack([np.zeros(3), R.flatten()])
        self.sample_trunk.value(pos)
        self.trunkTask.setReference(self.sample_trunk)
        
    def set_foot_3d_ref(self, pos, vel, acc, i):
        self.sample_feet_pos[i][:3] = pos
        self.sample_feet_vel[i][:3] = vel
        self.sample_feet_acc[i][:3] = acc
        self.sample_feet[i].value(self.sample_feet_pos[i])
        self.sample_feet[i].derivative(self.sample_feet_vel[i])
        self.sample_feet[i].second_derivative(self.sample_feet_acc[i])        
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