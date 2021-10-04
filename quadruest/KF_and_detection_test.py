import time
import numpy as np
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()
# from example_robot_data import loadANYmal
import ndcurves
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt

from contact_forces_estimator import ContactForcesEstimator
from filters import ImuLegKF, ImuLegCF

NOFEET = True
THRESH_FZ = 3  # minimum force to consider stable contact from estimated normal force (N)


if NOFEET:
    URDF_NAME = 'solo12_nofeet.urdf'
else:
    URDF_NAME = 'solo12.urdf'
path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/' + URDF_NAME
srdf = path + '/srdf/solo.srdf'
reference_config_q_name = 'standing'

LEGS = ['FL', 'FR', 'HL', 'HR']
if NOFEET:
    contact_frame_names = [leg+'_FOOT_TIP' for leg in LEGS]  # ......... -> solo12_nofeet
else:
    contact_frame_names = [leg+'_ANKLE' for leg in LEGS]  # # same thing as XX_FOOT but contained in pybullet -> solo12

controlled_joint_names = []
for leg in LEGS:
    controlled_joint_names += [leg + '_HAA', leg + '_HFE', leg + '_KFE']

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
model = robot.model
data = robot.data
pin.loadReferenceConfigurations(model, srdf, False)

# cs_file_dir = '/home/mfourmy/Documents/Phd_LAAS/phd_misc/centroidalkin/data/multicontact-api-master-examples/examples/quadrupeds/'
# cs_file_name = 'anymal_walk_WB.cs'

cs_file_dir = '/home/mfourmy/Documents/Phd_LAAS/data/trajs/'
# cs_file_name = 'solo_nomove.cs'
# cs_file_name = 'solo_sin_y_notrunk.cs'
cs_file_name = 'solo_sin_y_notrunk_Pyb_Nofeet.cs'
cs_file_path = cs_file_dir + cs_file_name

cs = ContactSequence()
cs.loadFromBinary(cs_file_path)

##############################################
# extract raw trajectory from contact sequence
##############################################
q_traj   = cs.concatenateQtrajectories()
dq_traj  = cs.concatenateDQtrajectories()
ddq_traj = cs.concatenateDDQtrajectories()
tau_traj = cs.concatenateTauTrajectories()
leg_name_lst = cs.getAllEffectorsInContact()
print('EE in contact')
print(leg_name_lst)
leg_cforces_traj_lst = [cs.concatenateContactForceTrajectories(leg_name) for leg_name in leg_name_lst]

min_ts = q_traj.min()
max_ts = q_traj.max()

dt = 1e-3  # discretization timespan
t_arr = np.arange(min_ts, max_ts, dt)
N = t_arr.shape[0]  # nb of discretized timestamps
# #####
# # for testing
# N = 1000
# t_arr = t_arr[:N]
# ####
q_arr   = np.array([q_traj(t) for t in t_arr])
dq_arr  = np.array([dq_traj(t) for t in t_arr])
ddq_arr = np.array([ddq_traj(t) for t in t_arr])
tau_arr = np.array([tau_traj(t) for t in t_arr])
leg_cforces_arr_lst = [np.array([leg_cforces_traj(t) for t in t_arr]) for leg_cforces_traj in leg_cforces_traj_lst]



###########################
# initialize the estimators
###########################
feet_id_lst = [robot.model.getFrameId(leg_name) for leg_name in leg_name_lst]
# initial state
robot.forwardKinematics(q_arr[0])
o_pl_lst = [robot.framePlacement(q_arr[0], leg_id, update_kinematics=False).translation for leg_id in feet_id_lst]

oRb = pin.Quaternion(q_arr[0,3:7].reshape((4,1))).toRotationMatrix()
o_vb = oRb @ dq_arr[0,:3]

cforces_est = ContactForcesEstimator(robot, feet_id_lst)

# define a prior for the FK state
x_init = np.concatenate((q_arr[0,:3], o_vb, o_pl_lst[0], o_pl_lst[1], o_pl_lst[2], o_pl_lst[3]))
std_p_prior = 0.01*np.ones(3)
std_v_prior = 0.01*np.ones(3)
std_pl0_prior = 0.1*np.ones(3)
std_pl1_prior = 0.1*np.ones(3)
std_pl2_prior = 0.1*np.ones(3)
std_pl3_prior = 0.1*np.ones(3)
std_prior = np.concatenate((std_p_prior, std_v_prior, std_pl0_prior, std_pl1_prior, std_pl2_prior, std_pl3_prior))

# noise applied to measurements
std_foot = 0.01
std_foot_vel = 0.01
std_acc = 0.1
std_wb = 0.001
std_qa = 0.001
std_dqa = 0.001
std_kin = 0.001
std_hfoot = 0.001

# filter noises
std_foot_kf = 0.01
std_foot_vel_kf = 0.01
std_acc_kf = 0.1
std_wb_kf = 0.000
std_qa_kf = 0.001
std_dqa_kf = 0.001
std_kin_kf = 0.001  # kinematic uncertainties
std_hfoot_kf = 0.001  # terrain height uncertainty -> roughness of the terrain

kf = ImuLegKF(robot, dt, feet_id_lst, x_init, std_prior, 
              std_foot_kf, std_foot_vel_kf, std_acc_kf, std_wb_kf, std_qa, std_dqa, std_kin_kf, std_hfoot_kf)

cf = ImuLegCF(robot, dt, feet_id_lst, x_init[:6])

# some useful recordings
x_arr_kf = np.zeros((N, x_init.shape[0]))
x_arr_cf = np.zeros((N, 6))
o_vb_arr = np.zeros((N, 3))
o_p_arr_gtr_lst = [np.zeros((N, 3)) for _ in range(4)]

for i in range(N):
    # define measurements
    oRb = pin.Quaternion(q_arr[i,3:7].reshape((4,1))).toRotationMatrix()

    o_acc = oRb @ (ddq_arr[i,:3] + np.cross(dq_arr[i,3:6], dq_arr[i,:3]))
    b_vb = dq_arr[i,0:3]  # not a measurement -> should come from the estimation
    b_wb = dq_arr[i,3:6]
    qa = q_arr[i,7:]
    dqa = dq_arr[i,6:]
    tau_joint = tau_arr[i,:]

    o_acc += np.random.normal(np.zeros(3), std_acc)
    b_wb += np.random.normal(np.zeros(3), std_wb)

    # compute forces --> cheating because uses b_bv ground truth
    o_forces = cforces_est.compute_contact_forces(qa, dqa, oRb, b_vb, b_wb, o_acc, tau_joint)
    feet_in_contact_ids = [cid for fz, cid in zip(o_forces[2,:], feet_id_lst) if fz > THRESH_FZ]  # simple contact detection

    # Kalman Filter
    t1 = time.time()
    kf.propagate(o_acc, feet_id_lst)
    kf.correct(qa, dqa, oRb, b_wb, feet_in_contact_ids)  # ! copy because mutated inside correct, to change
    # print((time.time() - t1)*1000, ' ms for propa+corr')
    x_arr_kf[i,:] = kf.get_state()

    # Complementary filter
    cf.update_state(o_acc, qa, dqa, b_wb, oRb, feet_in_contact_ids)
    x_arr_cf[i,:] = cf.get_state()

    # retrieve some ground truth
    o_vb_arr[i,:] = oRb @ dq_arr[i,:3]
    robot.forwardKinematics(q_arr[i,:])
    for foot_nb in range(4):
        o_p_arr_gtr_lst[foot_nb][i,:] = robot.framePlacement(q_arr[i,:], feet_id_lst[foot_nb], update_kinematics=False).translation



# Base error
p_err_KF_x = x_arr_kf[:,0] - q_arr[:,0]
p_err_KF_y = x_arr_kf[:,1] - q_arr[:,1]
p_err_KF_z = x_arr_kf[:,2] - q_arr[:,2]
v_err_KF_x = x_arr_kf[:,3] - o_vb_arr[:,0]
v_err_KF_y = x_arr_kf[:,4] - o_vb_arr[:,1]
v_err_KF_z = x_arr_kf[:,5] - o_vb_arr[:,2]

p_err_CF_x = x_arr_cf[:,0] - q_arr[:,0]
p_err_CF_y = x_arr_cf[:,1] - q_arr[:,1]
p_err_CF_z = x_arr_cf[:,2] - q_arr[:,2]
v_err_CF_x = x_arr_cf[:,3] - o_vb_arr[:,0]
v_err_CF_y = x_arr_cf[:,4] - o_vb_arr[:,1]
v_err_CF_z = x_arr_cf[:,5] - o_vb_arr[:,2]


def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2))

print('Position RMSE')
print('p_err_CF_x: ', rmse(p_err_CF_x))
print('p_err_CF_y: ', rmse(p_err_CF_y))
print('p_err_CF_z: ', rmse(p_err_CF_z))
print('p_err_KF_x: ', rmse(p_err_KF_x))
print('p_err_KF_y: ', rmse(p_err_KF_y))
print('p_err_KF_z: ', rmse(p_err_KF_z))

print()
print('Velocity RMSE')
print('v_err_CF_x: ', rmse(v_err_CF_x))
print('v_err_CF_y: ', rmse(v_err_CF_y))
print('v_err_CF_z: ', rmse(v_err_CF_z))
print('v_err_KF_x: ', rmse(v_err_KF_x))
print('v_err_KF_y: ', rmse(v_err_KF_y))
print('v_err_KF_z: ', rmse(v_err_KF_z))


#############
# ERROR plots
#############


plt.figure('Kalman position errors')
plt.plot(t_arr, p_err_KF_x, 'r', label='x err')
plt.plot(t_arr, p_err_KF_y, 'g', label='y err')
plt.plot(t_arr, p_err_KF_z, 'b', label='z err')
plt.legend()

plt.figure('Kalman velocity errors')
plt.plot(t_arr, v_err_KF_x, 'r', label='vx err')
plt.plot(t_arr, v_err_KF_y, 'g', label='vy err')
plt.plot(t_arr, v_err_KF_z, 'b', label='vz err')
plt.legend()

plt.figure('Complementary position errors')
plt.plot(t_arr, p_err_CF_x, 'r', label='x err')
plt.plot(t_arr, p_err_CF_y, 'g', label='y err')
plt.plot(t_arr, p_err_CF_z, 'b', label='z err')
plt.legend()

plt.figure('Complementary velocity errors')
plt.plot(t_arr, v_err_CF_x, 'r', label='vx err')
plt.plot(t_arr, v_err_CF_y, 'g', label='vy err')
plt.plot(t_arr, v_err_CF_z, 'b', label='vz err')
plt.legend()


# feet position error
o_p_arr_lst = [x_arr_kf[:,6:9], x_arr_kf[:,9:12], x_arr_kf[:,12:15], x_arr_kf[:,15:18]]

NUMBER_OF_LEGS_TO_PLOT = 1
for foot_nb in range(NUMBER_OF_LEGS_TO_PLOT):
    plt.figure('Feet position error ' + leg_name_lst[foot_nb])
    plt.plot(t_arr, o_p_arr_lst[foot_nb][:,0] - o_p_arr_gtr_lst[foot_nb][:,0], 'r')
    plt.plot(t_arr, o_p_arr_lst[foot_nb][:,1] - o_p_arr_gtr_lst[foot_nb][:,1], 'g')
    plt.plot(t_arr, o_p_arr_lst[foot_nb][:,2] - o_p_arr_gtr_lst[foot_nb][:,2], 'b')


############
# traj plots
############

# plt.figure('position trajs')
# plt.subplot(3,1,1)
# plt.plot(t_arr, x_arr_kf[:,0], 'r:', label='x est')
# plt.plot(t_arr, q_arr [:,0], 'r',  label='x gtr')
# plt.subplot(3,1,2)
# plt.plot(t_arr, x_arr_kf[:,1], 'g:', label='y est')
# plt.plot(t_arr, q_arr [:,1], 'g',  label='y gtr')
# plt.subplot(3,1,3)
# plt.plot(t_arr, x_arr_kf[:,1], 'b:', label='z est')
# plt.plot(t_arr, q_arr [:,1], 'b',  label='z gtr')
# plt.legend()

# plt.figure('velocity trajs')
# plt.subplot(3,1,1)
# plt.plot(t_arr, x_arr_kf   [:,3], 'r:', label='vx est')
# plt.plot(t_arr, o_vb_arr[:,0], 'r',  label='vx gtr')
# plt.subplot(3,1,2)
# plt.plot(t_arr, x_arr_kf   [:,4], 'g:', label='vy est')
# plt.plot(t_arr, o_vb_arr[:,1], 'g',  label='vy gtr')
# plt.subplot(3,1,3)
# plt.plot(t_arr, x_arr_kf   [:,5], 'b:', label='vz est')
# plt.plot(t_arr, o_vb_arr[:,2], 'b',  label='vz gtr')
# plt.legend()



plt.show()
