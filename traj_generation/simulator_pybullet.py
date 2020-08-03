import numpy as np
import time

import pybullet as pyb  # Pybullet server
import pybullet_data
import pinocchio as pin


class SimulatorPybullet:

    def __init__(self, dt, q_init, nqa, gravity=[0,0,-9.81]):

        # Start the client for PyBullet
        physicsClient = pyb.connect(pyb.GUI)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

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
        self.robotId = pyb.loadURDF(
            "solo12.urdf", robotStartPos, robotStartOrientation)

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
        self.qmes12 = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

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
                                    cameraTargetPosition=[pyb_sim.qmes12[0, 0], pyb_sim.qmes12[1, 0], 0.0])

        # Vector that contains torques
        jointTorques = 0.1 * np.sin(2 * np.pi * i * 0.001 * np.ones((12, 1)))

        # Set control torque for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                    controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Compute one step of simulation
        pyb.stepSimulation()

        # Wait a bit
        time.sleep(0.001)
