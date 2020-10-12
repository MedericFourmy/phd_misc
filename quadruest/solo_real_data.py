import time
import numpy as np
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()
# from example_robot_data import loadANYmal
import curves
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt

from contact_forces_estimator import ContactForcesEstimator
from filters import ImuLegKF, ImuLegCF

from data_readers import read_data_files_mpi, read_data_file_laas, shortened_arr_dic

SOLO12 = True
THRESH_FZ = 4  # minimum force to consider stable contact from estimated normal force (N)

if SOLO12:
    URDF_NAME = 'solo12.urdf'
else:
    URDF_NAME = 'solo.urdf'

path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/' + URDF_NAME
# srdf = path + '/srdf/solo.srdf'
# reference_config_q_name = 'standing'

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
model = robot.model
data = robot.data

# LEGS = ['FL', 'FR', 'HL', 'HR']
LEGS = ['FL', 'FR', 'HR'] # !!! remove HL because of faulty leg in Logs_09_10_20_soir dataset 
nb_feet = len(LEGS)
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

# Base to IMU transformation
# b_p_bi = np.zeros(3)
b_p_bi = np.array([0.1163, 0.0, 0.02])
b_q_i  = np.array([0, 0, 0, 1])
b_T_i = pin.SE3((pin.Quaternion(b_q_i.reshape((4,1)))).toRotationMatrix(), b_p_bi)
b_R_i = b_T_i.rotation
i_R_b = b_R_i.T
b_p_bi = b_T_i.translation

# measurements to be used in KF update
# MEASUREMENTS = (0,0,0)  # nothing happens
# MEASUREMENTS = (1,0,0)  # only kin
# MEASUREMENTS = (0,1,0)  # only diff kin
MEASUREMENTS = (1,1,0)  # all kinematics
# MEASUREMENTS = (1,1,1)  # all kinematics + foot height

##############################################
# extract raw trajectory from data file
##############################################
dt = 1e-3

# # folder = "data/solo12_standing_still_2020-10-01_14-22-52/2020-10-01_14-22-52/"
# # folder = "data/solo12_com_oscillation_2020-10-01_14-22-13/2020-10-01_14-22-13/"
# folder = "data/solo12_stamping_2020-09-29_18-04-37/2020-09-29_18-04-37/"
# # arr_dic = read_data_files_mpi(folder, dt)  # if default format
# arr_dic = read_data_files_mpi(folder, dt, delimiter=',')  # with "," delimiters

DATA_FOLDER_RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/'
DATA_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/'
# data_file = 'data.npz'
# data_file = 'data_2020_10_08_09_50_Walking_Novicon.npz'
# data_file = 'data_2020_10_08_10_04_StandingStill.npz'
# data_file = 'data_2020_10_09_16_10_Stamping.npz'
# data_file = 'data_2020_10_09_16_12_SinTraj.npz'

# 18:58 Fixe 4 stance phase (20s)
# 19:00 Rotation 4 stance phase (30s)
# 19:02 Mouvement avant arrière, rotation, rotation, mvt bas haut, roll (30s)
# 19:03 Replay sin wave
# 19:05 Replay stamping
# 19:06 Marche 0.32 (30s)
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_18_58.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_00.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_02.npz'
data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_03.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_05.npz'
# data_file = 'Logs_09_10_20_soir/data_2020_10_09_19_06.npz'

arr_dic = read_data_file_laas(DATA_FOLDER+data_file, dt)
# arr_dic = shortened_arr_dic(arr_dic, 2000)

t_arr = arr_dic['t']
N = len(t_arr)

###########################
# initialize the estimators
###########################
# initial state
# position: 0,0,0
# orientation: imu orientation -> HYP oRi=oRb
# velocity: 0,0,0 -> HYP robot does not move
o_p_oi = np.zeros(3)
o_q_i = arr_dic['o_q_i'][0,:]
o_v_oi = np.zeros(3)
qa = arr_dic['qa'][0,:]
q = np.hstack([o_p_oi, o_q_i, qa])

oRi = pin.Quaternion(o_q_i.reshape((4,1))).toRotationMatrix()

robot.forwardKinematics(q)
o_p_ol_lst = [robot.framePlacement(q, leg_id, update_kinematics=False).translation for leg_id in contact_frame_ids]


cforces_est = ContactForcesEstimator(robot, contact_frame_ids)

# define a prior for the FK state
x_init = np.concatenate((o_p_oi, o_v_oi, *o_p_ol_lst))

# prior covariances
std_p_prior = 0.01*np.ones(3)
std_v_prior = 1*np.ones(3)
std_pl_priors = 0.1*np.ones(3*nb_feet)
std_prior = np.concatenate((std_p_prior, std_v_prior, std_pl_priors))

# filter noises
std_kf_dic = {
    'std_foot': 0.1,# process noise on foot dynamics when in contact -> raised when stable contact interuption
    'std_acc': 0.1,# noise on linear acceleration measurements
    'std_wb': 0.001,# noise on angular velocity measurements
    'std_qa': 0.05,# noise on joint position measurements
    'std_dqa': 0.05,# noise on joint velocity measurements
    'std_kin': 0.01,# kinematic uncertainties
    'std_hfoot': 0.01,# terrain height uncertainty -> roughness of the terrain
} 

kf = ImuLegKF(robot, dt, contact_frame_ids, x_init, b_T_i, std_prior, std_kf_dic)
cf = ImuLegCF(robot, dt, contact_frame_ids, x_init[:6], b_T_i)

# some useful recordings
x_arr_kf = np.zeros((N, x_init.shape[0]))
x_arr_cf = np.zeros((N, 6))
i_v_oi_kf_arr = np.zeros((N, 3))
i_v_oi_cf_arr = np.zeros((N, 3))
f_sum_arr = np.zeros((N,3))
fz_arr = np.zeros((N,nb_feet))
feets_in_contact_arr= np.zeros((N,nb_feet))
contact_offset_viz = np.array([0.1, -0.1, 0.05, -0.05])[:nb_feet]

# base estimates for post processing
o_R_b_arr = np.zeros((N,3,3))
o_q_b_arr = np.zeros((N,4))
o_p_ob_kf_arr = np.zeros((N,3))
o_v_ob_kf_arr = np.zeros((N,3))
o_p_ob_cf_arr = np.zeros((N,3))
o_v_ob_cf_arr = np.zeros((N,3))

delays = np.zeros(N)
for i in range(N):
    # define measurements
    o_R_i = arr_dic['o_R_i'][i,:]  # retrieve IMU pose estimation
    o_a_oi = arr_dic['o_a_oi'][i,:] # retrieve IMU linear acceleration estimation

    i_v_oi = o_R_i.T @ o_v_oi  # again, HYP oRi=oRb
    i_omg_oi = arr_dic['i_omg_oi'][i,:]
    qa = arr_dic['qa'][i,:]
    dqa = arr_dic['dqa'][i,:]
    tau = arr_dic['tau'][i,:]

    # compute forces
    o_forces = cforces_est.compute_contact_forces(qa, dqa, o_R_i, i_v_oi, i_omg_oi, o_a_oi, tau)
    f_sum_arr[i,:] = sum(o_forces)
    fz_arr[i,:] = o_forces[:,2]
    # print('fz: ', fz_lst)
    feets_in_contact = [fz > THRESH_FZ for fz in o_forces[:,2]]  # simple contact detection

    feets_in_contact_arr[i,:] = np.array(feets_in_contact) + contact_offset_viz

    # Kalman Filter
    t1 = time.time()
    kf.propagate(o_a_oi, feets_in_contact)
    kf.correct(qa, dqa, o_R_i, i_omg_oi, feets_in_contact, measurements=MEASUREMENTS)
    delay = (time.time() - t1)*1000
    delays[i] = delay
    x_arr_kf[i,:] = kf.get_state()

    # Complementary filter
    cf.update_state(o_a_oi, qa, dqa, i_omg_oi, o_R_i, feets_in_contact)
    x_arr_cf[i,:] = cf.get_state()

    # update current vel estimation (for force reconstruction)
    o_v_oi = x_arr_kf[i,3:6]
    i_v_oi_kf_arr[i,:] = o_R_i.T @ x_arr_kf[i,3:6] 
    i_v_oi_cf_arr[i,:] = o_R_i.T @ x_arr_cf[i,3:6] 

    o_R_b_arr[i,:] = o_R_i @ i_R_b
    o_q_b_arr[i,:] = pin.Quaternion(o_R_b_arr[i,:]).coeffs()
    o_p_ob_kf_arr[i,:] = x_arr_kf[i,0:3] + o_R_b_arr[i,:] @ b_p_bi
    o_v_ob_kf_arr[i,:] = x_arr_kf[i,3:6] + o_R_b_arr[i,:] @ np.cross(i_omg_oi, b_p_bi)
    o_p_ob_cf_arr[i,:] = x_arr_cf[i,0:3] + o_R_b_arr[i,:] @ b_p_bi
    o_v_ob_cf_arr[i,:] = x_arr_cf[i,3:6] + o_R_b_arr[i,:] @ np.cross(i_omg_oi, b_p_bi)


# data to copy
res_arr_dic = {}
copy_lst = ['t', 'w_v_wm', 'm_v_wm', 'w_q_m', 'o_R_i', 'w_p_wm', 'i_omg_oi']
for k in copy_lst:
    res_arr_dic[k] = arr_dic[k]
# add estimated data
res_arr_dic['o_R_b'] =     o_R_b_arr
res_arr_dic['o_q_b'] =     o_q_b_arr
res_arr_dic['o_p_ob_kf'] = o_p_ob_kf_arr
res_arr_dic['o_v_ob_kf'] = o_v_ob_kf_arr
res_arr_dic['o_p_ob_cf'] = o_p_ob_cf_arr
res_arr_dic['o_v_ob_cf'] = o_v_ob_cf_arr


np.savez(DATA_FOLDER_RESULTS+data_file, **res_arr_dic)
print(DATA_FOLDER_RESULTS+data_file, ' saved')

# PLOT DELAYS
plt.figure('KF propa+correct time (ms)')
plt.plot(t_arr, delays)

# contacts
plt.figure('Normal forces')
for i in range(nb_feet):
    plt.plot(t_arr, fz_arr[:,i], label='fz '+contact_frame_names[i])
plt.legend()

plt.figure('Total forces world frame')
plt.plot(t_arr, f_sum_arr[:,0], label='f sum x')
plt.plot(t_arr, f_sum_arr[:,1], label='f sum y')
plt.plot(t_arr, f_sum_arr[:,2], label='f sum z')
plt.legend()

plt.figure('Contacts')
for i in range(nb_feet):
    plt.plot(t_arr, feets_in_contact_arr[:,i], '.', label=contact_frame_names[i], markersize=1)
plt.legend()

plt.show()
