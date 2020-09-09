import numpy as np
import time

import pybullet as pyb  # Pybullet server
import pybullet_data
import pinocchio as pin


class SimulatorPybullet:

    def __init__(self, dt, q_init, nqa, pinocchio_robot, joint_names, endeff_names, gravity=[0,0,-9.81]):
        
        
        # Start the client for PyBullet
        physicsClient = pyb.connect(pyb.GUI)
        # pyb.GUI for graphical version
        # pyb.DIRECT for non-graphical version

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
            "solo12.urdf", robotStartPos, robotStartOrientation, flags=flags)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        pyb.setJointMotorControlArray(self.robotId, 
                                      jointIndices=self.revoluteJointIndices, 
                                      controlMode=pyb.VELOCITY_CONTROL,
                                      targetVelocities=[
                                          0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, q_a)

        # Enable torque control for revolute joints
        jointTorques = [0.0]*len(self.revoluteJointIndices)
        pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        
        bullet_joint_map = {}
        for ji in range(pyb.getNumJoints(self.robotId)):
            bullet_joint_map[pyb.getJointInfo(self.robotId, ji)[1].decode('UTF-8')] = ji

        self.bullet_joint_ids = np.array([bullet_joint_map[name] for name in joint_names])
        self.pinocchio_joint_ids = np.array([pinocchio_robot.model.getJointId(name) for name in joint_names])


        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = [bullet_joint_map[name] for name in endeff_names]
        self.pinocchio_endeff_ids = [pinocchio_robot.model.getFrameId(name) for name in endeff_names]

        # Set time step for the simulation
        pyb.setTimeStep(dt)

        # Change camera position
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-50, cameraPitch=-35,
                                       cameraTargetPosition=[0.0, 0.6, 0.0])

    def retrieve_pyb_data(self):
        """Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """

        # Retrieve data from the simulation
        self.jointStates = pyb.getJointStates(
            self.robotId, self.revoluteJointIndices)  # State of all joints
        self.baseState = pyb.getBasePositionAndOrientation(
            self.robotId)  # Position and orientation of the trunk
        self.baseVel = pyb.getBaseVelocity(
            self.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

        return 0

    # def get_force(self):
    #     """ Returns the force readings as well as the set of active contacts """
    #     active_contacts_frame_ids = []
    #     contact_forces = []

    #     # Get the contact model using the pyb.getContactPoints() api.
    #     def sign(x):
    #         if x >= 0:
    #             return 1.
    #         else:
    #             return -1.

    #     cp = pyb.getContactPoints()

    #     for ci in reversed(cp):
    #         contact_normal = ci[7]
    #         normal_force = ci[9]
    #         lateral_friction_direction_1 = ci[11]
    #         lateral_friction_force_1 = ci[10]
    #         lateral_friction_direction_2 = ci[13]
    #         lateral_friction_force_2 = ci[12]

    #         if ci[3] in self.bullet_endeff_ids:
    #             i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
    #         elif ci[4] in self.bullet_endeff_ids:
    #             i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
    #         else:
    #             continue

    #         if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
    #             continue

    #         active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
    #         force = np.zeros(6)

    #         force[:3] = normal_force * np.array(contact_normal) + \
    #                     lateral_friction_force_1 * np.array(lateral_friction_direction_1) + \
    #                     lateral_friction_force_2 * np.array(lateral_friction_direction_2)

    #         contact_forces.append(force)

    #     return active_contacts_frame_ids[::-1], contact_forces[::-1]



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
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                    controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Compute one step of simulation
        pyb.stepSimulation()

        # Wait a bit
        time.sleep(0.001)
