import numpy as np
import time

import pybullet as pyb  # Pybullet server
import pybullet_data
import pinocchio as pin


class SimulatorPybullet:

    def __init__(self, urdf_name, dt, q_init, nqa, pinocchio_robot, joint_names, contact_frame_names, guion=True, gravity=[0,0,-9.81]):
        """
        contact_frame_names: ! to work, contact_frame_names needs to be included in the joint frame names
        """
        
        # Start the client for PyBullet
        if guion:
            pyb.connect(pyb.GUI)
        else:
            pyb.connect(pyb.DIRECT)

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pyb.loadURDF("plane.urdf")

        # Set the gravity
        pyb.setGravity(*gravity)

        # Load Quadruped robot with init config (with containing freeflyer pose)
        robotStartPos = q_init[:3]
        robotStartOrientation = q_init[3:7]
        q_a = q_init[7:].reshape((nqa, 1))

        pyb.setAdditionalSearchPath(
            "/opt/openrobots/share/example-robot-data/robots/solo_description/robots")

        flags = pyb.URDF_USE_INERTIA_FROM_FILE  # Necessary?
        self.robotId = pyb.loadURDF(
            urdf_name, robotStartPos, robotStartOrientation, flags=flags)

        # get joint ids informations in pyb and pinocchio
        bullet_joint_map = {}
        for ji in range(pyb.getNumJoints(self.robotId)):
            bullet_joint_map[pyb.getJointInfo(self.robotId, ji)[1].decode('UTF-8')] = ji

        self.bullet_joint_ids = np.array([bullet_joint_map[name] for name in joint_names])
        self.pinocchio_joint_ids = np.array([pinocchio_robot.model.getJointId(name) for name in joint_names])

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.pyb_ct_frame_ids = [bullet_joint_map[name] for name in contact_frame_names]
        self.pin_ct_frame_ids = [pinocchio_robot.model.getFrameId(name) for name in contact_frame_names]

        # Disable default motor control for revolute joints
        pyb.setJointMotorControlArray(self.robotId, 
                                      jointIndices=self.bullet_joint_ids, 
                                      controlMode=pyb.VELOCITY_CONTROL,
                                      forces=[0.0 for _ in self.bullet_joint_ids])
                                      
        # Initialize the robot in a specific configuration
        pyb.resetJointStatesMultiDof(self.robotId, self.bullet_joint_ids, q_a)

        # Enable torque control for revolute joints
        jointTorques = [0.0]*len(self.bullet_joint_ids)
        pyb.setJointMotorControlArray(self.robotId, self.bullet_joint_ids,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Set time step for the simulation
        pyb.setTimeStep(dt)

        # Change camera position
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-50, cameraPitch=-35,
                                       cameraTargetPosition=[0.0, 0.6, 0.0])

    def retrieve_pyb_data(self):
        """
        Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """

        # Joint states
        jointStates = pyb.getJointStates(self.robotId, self.bullet_joint_ids)
        self.qa = np.array([state[0] for state in jointStates])
        self.va = np.array([state[1] for state in jointStates])

        # Position and orientation of the trunk
        self.w_p_b, self.w_quat_b = pyb.getBasePositionAndOrientation(self.robotId)
        self.w_p_b, self.w_quat_b = np.array(self.w_p_b), np.array(self.w_quat_b)
        self.w_R_b = pin.Quaternion(self.w_quat_b.reshape((4,1))).toRotationMatrix()

        # Velocity of the trunk -> world coordinates given by pyb
        self.w_v_b, self.w_omg_b = pyb.getBaseVelocity(self.robotId)
        self.w_v_b, self.w_omg_b = np.array(self.w_v_b), np.array(self.w_omg_b)
        self.b_v_b, self.b_omg_b = self.w_R_b.T @ self.w_v_b, self.w_R_b.T @ self.w_omg_b

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.q = np.hstack((self.w_p_b, self.w_quat_b, self.qa))
        self.v = np.hstack((self.b_v_b, self.b_omg_b,  self.va))

        return 0


if __name__ == '__main__':

    # Initialisation of the PyBullet simulator
    robotStartPos = np.array([0, 0, 0.235+0.0045])
    robotStartOrientation = np.array(pyb.getQuaternionFromEuler([0.0, 0.0, 0.0]))
    q_a = np.array([0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])  # straight_standing
    q_init = np.concatenate([robotStartPos, robotStartOrientation, q_a]) 
    pyb_sim = SimulatorPybullet(dt=0.001, q_init=q_init, nqa=12)

    for i in range(5000):

        # Get position/orientation of the base and angular position of actuators
        pyb_sim.retrieve_pyb_data()

        # Center the camera on the current position of the robot
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-50, cameraPitch=-39.9,
                                    cameraTargetPosition=[pyb_sim.qmes[0, 0], pyb_sim.qmes[1, 0], 0.0])

        # Vector that contains torques
        jointTorques = 0.1 * np.sin(2 * np.pi * i * 0.001 * np.ones((12, 1)))

        # Set control torque for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.bullet_joint_ids,
                                    controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Compute one step of simulation
        pyb.stepSimulation()

        # Wait a bit
        time.sleep(0.001)
