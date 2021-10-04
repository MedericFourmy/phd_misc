
import time
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import c_double
import numpy as np
import vrpn

import pinocchio
from pinocchio.utils import se3ToXYZQUAT
from pinocchio.explog import log


FREQ_MOCAP = 120


class VRPNClient():
    def __init__(self, ip="192.168.101.93", body_id="robot"):
        # shared c_double array
        self.shared_bodyPosition = Array(c_double, 3, lock=False)
        self.shared_bodyVelocity = Array(c_double, 3, lock=False)
        self.shared_bodyOrientationQuat = Array(c_double, 4, lock=False)
        self.shared_bodyOrientationMat9 = Array(c_double, 9, lock=False)
        self.shared_bodyAngularVelocity = Array(c_double, 3, lock=False)
        self.shared_timestamp = Value(c_double, lock=False)
        args = (ip, body_id, self.shared_bodyPosition, self.shared_bodyVelocity,
                self.shared_bodyOrientationQuat, self.shared_bodyOrientationMat9,
                self.shared_bodyAngularVelocity, self.shared_timestamp)
        self.p = Process(target=self.vrpn_process, args=args)
        self.p.start()


    def stop(self):
        self.p.terminate()
        self.p.join()

    def getPosition(self):
        return np.array([self.shared_bodyPosition[0],
                         self.shared_bodyPosition[1],
                         self.shared_bodyPosition[2]])

    def getVelocity(self):
        return np.array([self.shared_bodyVelocity[0],
                         self.shared_bodyVelocity[1],
                         self.shared_bodyVelocity[2]])

    def getAngularVelocity(self):
        return np.array([self.shared_bodyAngularVelocity[0],
                         self.shared_bodyAngularVelocity[1],
                         self.shared_bodyAngularVelocity[2]])

    def getOrientationMat9(self):
        return np.array([[self.shared_bodyOrientationMat9[0], self.shared_bodyOrientationMat9[1], self.shared_bodyOrientationMat9[2]],
                         [self.shared_bodyOrientationMat9[3], self.shared_bodyOrientationMat9[4],
                             self.shared_bodyOrientationMat9[5]],
                         [self.shared_bodyOrientationMat9[6], self.shared_bodyOrientationMat9[7], self.shared_bodyOrientationMat9[8]]])

    def getOrientationQuat(self):
        return np.array([self.shared_bodyOrientationQuat[0],
                         self.shared_bodyOrientationQuat[1],
                         self.shared_bodyOrientationQuat[2],
                         self.shared_bodyOrientationQuat[3]])

    def vrpn_process(self, ip, body_id, shared_bodyPosition, shared_bodyVelocity,
                         shared_bodyOrientationQuat, shared_bodyOrientationMat9,
                         shared_bodyAngularVelocity, shared_timestamp):
        print('VRPN process!')
        ''' This will run on a different process'''
        shared_timestamp.value = -1

        def callback(userdata, data):
            """
            Callback function that is called everytime a data packet arrives from VRPN protocol.
            The buffer if depiled every time "mainloop() is called.
            """
            timestamp = data['time'].timestamp()
            position = np.array(data['position'])
            quaternion = pinocchio.Quaternion(np.array(data['quaternion']))
            rotation = quaternion.toRotationMatrix()

            # Get the last position and Rotation matrix from the shared memory.
            last_position = np.array([shared_bodyPosition[i] for i in range(3)])
            last_rotation = np.array([shared_bodyOrientationMat9[i] for i in range(9)]).reshape((3,3))

            # Get the position, Rotation matrix and Quaternion
            shared_bodyPosition[0] = position[0]
            shared_bodyPosition[1] = position[1]
            shared_bodyPosition[2] = position[2]
            shared_bodyOrientationQuat[0] = quaternion[0]
            shared_bodyOrientationQuat[1] = quaternion[1]
            shared_bodyOrientationQuat[2] = quaternion[2]
            shared_bodyOrientationQuat[3] = quaternion[3]

            rotation_flat = rotation.flatten()
            shared_bodyOrientationMat9[0] = rotation_flat[0]
            shared_bodyOrientationMat9[1] = rotation_flat[1]
            shared_bodyOrientationMat9[2] = rotation_flat[2]
            shared_bodyOrientationMat9[3] = rotation_flat[3]
            shared_bodyOrientationMat9[4] = rotation_flat[4]
            shared_bodyOrientationMat9[5] = rotation_flat[5]
            shared_bodyOrientationMat9[6] = rotation_flat[6]
            shared_bodyOrientationMat9[7] = rotation_flat[7]
            shared_bodyOrientationMat9[8] = rotation_flat[8]

            # Compute world velocity.
            if (shared_timestamp.value == -1):
                shared_bodyVelocity[0] = 0
                shared_bodyVelocity[1] = 0
                shared_bodyVelocity[2] = 0
                shared_bodyAngularVelocity[0] = 0.0
                shared_bodyAngularVelocity[1] = 0.0
                shared_bodyAngularVelocity[2] = 0.0
            else:
                dt = timestamp - shared_timestamp.value
                shared_bodyVelocity[0] = (position[0] - last_position[0])/dt
                shared_bodyVelocity[1] = (position[1] - last_position[1])/dt
                shared_bodyVelocity[2] = (position[2] - last_position[2])/dt
                bodyAngularVelocity = log(last_rotation.T @ rotation)/dt
                shared_bodyAngularVelocity[0] = bodyAngularVelocity[0]
                shared_bodyAngularVelocity[1] = bodyAngularVelocity[1]
                shared_bodyAngularVelocity[2] = bodyAngularVelocity[2]

            shared_timestamp.value = timestamp

        tracker = vrpn.receiver.Tracker(body_id+"@"+ip)
        tracker.register_change_handler("position", callback, "position")

        while 1:
            tracker.mainloop()
            time.sleep(1/FREQ_MOCAP)



def exampleOfUse():
    import time
    qc = VRPNClient(ip="192.168.101.93", body_id="robot")
    for i in range(300):
        # print(chr(27) + "[2J")
        print("position:         ", qc.getPosition())
        print("quaternion:       ", qc.getOrientationQuat())
        print("linear velocity:  ", qc.getVelocity())
        print("angular velocity: ", qc.getAngularVelocity())
        time.sleep(0.3)
        # time.sleep(1/120)
    print("killme!")



if __name__ == "__main__":
    exampleOfUse()
