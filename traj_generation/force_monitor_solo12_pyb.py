# coding: utf8

"""
Written by Pierre-Alexandre
"""

import numpy as np
import pybullet as pyb


class ForceMonitor:

    def __init__(self, robotId, planeId):

        self.lines = []
        self.robotId = robotId
        self.planeId = planeId

    def getContactPoint(self, contactPoints):
        """ Sort contacts points as there should be only one contact per foot
            and sometimes PyBullet detect several of them. There is one contact
            with a non zero force and the others have a zero contact force
        """

        for i in range(0, len(contactPoints)):
            # There may be several contact points for each foot but only one of them as a non zero normal force
            if (contactPoints[i][9] != 0):
                return contactPoints[i]

        # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen)
        return 0

    def get_contact_forces(self, display=False):
        # Front left foot, Front right foot, Hind left  foot, Hind right foot
        link_idx = [3, 7, 11, 15][::-1]
        # contact_frame_names = ['HR_ANKLE', 'HL_ANKLE', 'FR_ANKLE', 'FL_ANKLE']  # same thing as XX_FOOT but contained in pybullet

        # forces are exactly zero when contact not active, also storing the contact point in world
        forces = {idx: {'f': np.zeros(3), 'wpc': np.zeros(3)} for idx in link_idx}  

        i_line = 0
        K = 0.02
        # Info about contact points with the ground
        for idx in link_idx:
            ct = pyb.getContactPoints(self.robotId, self.planeId, linkIndexA=idx)
            # Sort contacts points to get only one contact per foot
            ct = self.getContactPoint(ct)
            if not isinstance(ct, int):
                start = [ct[6][0], ct[6][1], ct[6][2]+0.04]
                end = [ct[6][0], ct[6][1], ct[6][2]+0.04]
                
                # extracting force in world frame
                for idir in range(0, 3):
                    forces[idx]['f'][idir] = (ct[9] * ct[7][idir] + ct[10] *
                                                  ct[11][idir] + ct[12] * ct[13][idir])
                    end[idir] += K * forces[idx]['f'][idir]

                # extracting contact point in world frame
                forces[idx]['wpc'][:] = ct[6]
                
                if display:
                    if (i_line+1) > len(self.lines):  # If not enough existing lines in line storage a new item is created
                        lineID = pyb.addUserDebugLine(start, end, lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
                        self.lines.append(lineID)
                    else:  # If there is already an existing line item we modify it (to avoid flickering)
                        self.lines[i_line] = pyb.addUserDebugLine(start, end, lineColorRGB=[
                            1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_line])
                    i_line += 1

        if display:
            for i_zero in range(i_line, len(self.lines)):
                self.lines[i_zero] = pyb.addUserDebugLine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], lineColorRGB=[
                    1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_zero])
        
        return forces