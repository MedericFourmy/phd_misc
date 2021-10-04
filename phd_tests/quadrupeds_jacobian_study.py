import pickle
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import seaborn as sns

from example_robot_data import loadANYmal
import ndcurves
from multicontact_api import ContactSequence



examples_dir = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/centroidalkin/data/multicontact-api-master-examples/examples/quadrupeds/'
file_name = 'anymal_walk_WB.cs'


cs = ContactSequence()
cs.loadFromBinary(examples_dir + file_name) 

q0 = cs.concatenateQtrajectories()(0)
contact_frames = cs.getAllEffectorsInContact()


robot = loadANYmal()
rmodel = robot.model
rdata = robot.data
contact_frame_ids = [rmodel.getFrameId(l) for l in contact_frames]


def compute_joint_jac(robot, q, cf_ids, world_frame=True):
    Jlinvel = np.zeros((12, robot.nv))
    for i, frame_id in enumerate(cf_ids):
        if world_frame:
            oTl = robot.framePlacement(q, frame_id, update_kinematics=False)
            Jlinvel[3*i:3*(i+1),:] = oTl.rotation @ robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in world coord
        else: 
            Jlinvel[3*i:3*(i+1),:] = robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in local coord
    return Jlinvel



Jlinvel = compute_joint_jac(robot, q0, contact_frame_ids)
Jlinvel_minus_freflyer = Jlinvel[:,6:]

# rank of jacobian and subjacobians
Jlinvel_rank = np.linalg.matrix_rank(Jlinvel)
Jlinvel_minus_freeflyer_rank = np.linalg.matrix_rank(Jlinvel_minus_freflyer)
leg_rank = np.linalg.matrix_rank(Jlinvel_minus_freflyer[0:3,0:3])
print('Jlinvel_rank: ', Jlinvel_rank)
print('Jlinvel_minus_freeflyer_rank: ', Jlinvel_minus_freeflyer_rank)
print('leg_rank: ', leg_rank)


plt.figure()
mask = np.zeros_like(Jlinvel)
mask[Jlinvel == 0] = True
sns.heatmap(Jlinvel, mask=mask, linewidths=.5)

plt.figure()
mask = np.zeros_like(Jlinvel_minus_freflyer)
mask[Jlinvel_minus_freflyer == 0] = True
sns.heatmap(Jlinvel_minus_freflyer, mask=mask, linewidths=.5)
plt.show()
