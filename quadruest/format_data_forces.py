import sys
import numpy as np
import pandas as pd
from numpy.core.defchararray import array
from scipy import signal
import pinocchio as pin
from scipy.ndimage.measurements import label
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib import pyplot as plt
from data_readers import read_data_file_laas, read_data_files_mpi, shortened_arr_dic
from contact_forces_estimator import ContactForcesEstimator
from example_robot_data import load



def interpolate_mocap(arr_dic):
    """
    Interpolate mocap nan data data using linear interpolation for
    the translation part and slerp for the quaternion 
    """

    t_arr = arr_dic['t']
    p_arr = arr_dic['w_p_wm']
    q_arr = arr_dic['w_q_m']

    idx_nnan = np.where(~np.isnan(p_arr[:,0]))[0]

    t_arr_nnan = t_arr[idx_nnan]
    p_arr_nnan = p_arr[idx_nnan]
    q_arr_nnan = q_arr[idx_nnan]

    # position linear interpolation
    p_arr_interp =  pd.DataFrame(p_arr).interpolate().to_numpy()

    # quaternion slerp
    r_arr_nnan = R.from_quat(q_arr_nnan)
    slerp = Slerp(t_arr_nnan, r_arr_nnan)
    r_interp = slerp(t_arr)
    q_arr_interp = r_interp.as_quat()

    arr_dic['w_p_wm'] = p_arr_interp
    arr_dic['w_q_m'] = q_arr_interp




# New experiments with Solo handled in the air for Mocap+IMU fusion
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_09.npz'  
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_10.npz'  
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_16.npz'  
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_17.npz'  
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_29.npz'         
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_25.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_26.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_30.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_32.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_31.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_54.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_56.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_57.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_59.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_16_03.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_24.npz'  # 5 minutes standing still
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_54.npz'  # //
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_05_19/data_2021_05_19_16_56.npz'  # //


# Sin trajs with calibration procedures
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_16_48.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_04.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_08.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_11.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_16.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMUKinematics_2021_06_11/data_2021_06_11_17_19.npz'


# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_36.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_38.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_Replay_2021_07_07/data_2021_07_07_13_39.npz'


# New experiments at IRI
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_bis.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_in_air_full.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21/solo_sin_back_down_rots_pointed_feet_ters.npz'

# New experiments at LAAS (with Joan)

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_pointed.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_sin_back_down_rots_on_planck_cleaner.npz'



# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_10s.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/LAAS_10_21/solo_in_air_1min.npz'

# IRI SECOND PART
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_1min.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_in_air_2min.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_19_10.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_bis.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_ter.npz'

# HIGHER
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_kp3.npz'  # not a zip file
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_kp2.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_kp1.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_planck_20_10_higher_2min_kp2_vib02.npz'

# OFFSETS
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_offset_calibration.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_bis.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_bzzz.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_sin_back_down_rots_ground_30_pybullet.npz'

# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_pointfeet_half_withstops.npz'

# WALKING TRAJECTORIES
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI.npz'
file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_stamping_IRI_bis.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_gait_10_10.npz'
# file_path = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/IRI_10_21_2nd/solo_gait_5_15.npz'

# PLANNED CONTACTS
arr_plan = np.load('/home/mfourmy/Documents/Phd_LAAS/data/trajs/solo_stamping.npz')


# OUT_FILE_NAME = file_path.split('.')[0]+'_calib_format.npz'
# OUT_FILE_NAME = file_path.split('.')[0]+'_move_format.npz'
OUT_FILE_NAME = file_path.split('.')[0]+'_format.npz'
SHOW = '--show' in sys.argv
SAVE = '--save' in sys.argv
if not (SHOW or SAVE):
    raise ValueError('--show or --save!')

THRESH_VIZ = 4


dt = 1e-3  # discretization timespan
# dt = 2e-3  # discretization timespan
arr_dic = read_data_file_laas(file_path, dt)
# interpolate_mocap(arr_dic)


# ################################
# # SHORTEN
# ###############
print(arr_dic['t'].shape)
print(arr_plan['contacts'].shape)
arr_dic = shortened_arr_dic(arr_dic, 200)
contacts = arr_plan['contacts'][203:]
contacts = contacts.astype('float64')
# ##################################



t_arr = arr_dic['t']
N = len(t_arr)


# mean the forces
# arr_dic['o_a_oi'][:,:] = np.mean(arr_dic['o_a_oi'], axis=0)
# arr_dic['imu_acc'][:,:] = np.mean(arr_dic['imu_acc'], axis=0)
# arr_dic['i_omg_oi'][:,:] = np.mean(arr_dic['i_omg_oi'], axis=0)
# arr_dic['qa'][:,:] = np.mean(arr_dic['qa'], axis=0)



# filter the kinematics with centered window to compute 1rst and 2nd derivatives
# dqa_filt_arr = signal.savgol_filter(arr_dic['qa'], window_length=25, polyorder=2, deriv=1, delta=dt, mode='mirror', axis=0)
# dqa_filt_arr = np.diff(arr_dic['qa'], axis=0)/dt
dqa_filt_arr = (arr_dic['qa'] - np.roll(arr_dic['qa'], 1, axis=0))/dt
dqa_filt_arr[0] = 0
ddqa_filt_arr = signal.savgol_filter(arr_dic['qa'], window_length=25, polyorder=2, deriv=2, delta=dt, mode='mirror', axis=0)

# import ipdb; ipdb.set_trace()

# compute angular acceleration
o_R_i_arr = arr_dic['o_R_i']
w_R_m_arr = arr_dic['w_R_m']
i_omg_oi_arr = arr_dic['i_omg_oi']
o_omg_i_arr = np.array([o_R_i@i_omg_oi for o_R_i, i_omg_oi in zip(o_R_i_arr, i_omg_oi_arr)])
o_domg_i_arr = signal.savgol_filter(o_omg_i_arr, window_length=25, polyorder=2, deriv=1, delta=dt, mode='mirror', axis=0)

i_domg_i_arr = np.array([o_R_i.T@o_domg_i for o_R_i, o_domg_i in zip(o_R_i_arr, o_domg_i_arr)])

rpy_imu_arr = np.array([pin.rpy.matrixToRpy(o_R_i) for o_R_i in o_R_i_arr])
rpy_mocap_arr = np.array([pin.rpy.matrixToRpy(R) for R in w_R_m_arr])



# Now compute the forces using the robot model
robot = load('solo12')
# ######################################
# # apt install problem of example_robot_data
# URDF_NAME = 'solo12.urdf'
# path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
# urdf = path + '/robots/' + URDF_NAME
# srdf = path + '/srdf/solo.srdf'
# robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
# model = robot.model
# data = robot.data
# pin.loadReferenceConfigurations(model, srdf, False)
# ######################################


robot.model.gravity.linear = np.array([0, 0, -9.806])
LEGS = ['FL', 'FR', 'HL', 'HR']
nb_feet = len(LEGS)
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

cforces_est = ContactForcesEstimator(robot, contact_frame_ids, dt)

o_forces_arr = np.zeros((N,12))
l_forces_arr = np.zeros((N,12))
o_forces_sum = np.zeros((N,3))
detect_arr = np.zeros((N,4))
for i in range(N):

    qa = arr_dic['qa'][i,:]
    dqa = dqa_filt_arr[i,:]
    ddqa = ddqa_filt_arr[i,:]*0
    o_R_i = o_R_i_arr[i,:,:]
    i_omg_oi = i_omg_oi_arr[i,:]
    i_domg_i = i_domg_i_arr[i,:]
    o_a_oi = arr_dic['o_a_oi'][i,:]
    tauj = arr_dic['tau'][i,:]

    o_forces = cforces_est.compute_contact_forces2(qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_i, o_a_oi, tauj, world_frame=True)
    l_forces = cforces_est.compute_contact_forces2(qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_i, o_a_oi, tauj, world_frame=False)

    o_forces_sum[i,:] = np.sum(o_forces, axis=0)

    o_forces_arr[i,:] = o_forces.flatten()
    l_forces_arr[i,:] = l_forces.flatten()

    detect_arr[i,:] = (o_forces[:,2] > THRESH_VIZ)

# store forces
arr_dic['o_forces'] = o_forces_arr
arr_dic['l_forces'] = l_forces_arr
# arr_dic['l_forces'] = np.zeros(12)

arr_dic['contacts'] = contacts



arr_dic['dqa'] = dqa_filt_arr


if SAVE:
    np.savez(OUT_FILE_NAME, **arr_dic)
    print(OUT_FILE_NAME, ' saved')

# q = robot.model.referenceConfigurations['standing']
q = robot.model.referenceConfigurations['standing']
robot.com(q)
m = robot.data.mass[0]
g = robot.model.gravity


o_forces_sum[:,:] = np.mean(o_forces_sum, axis=0)


plt.figure('o f sum')
plt.plot(t_arr, o_forces_sum[:,0], 'r.', markersize=1)
plt.plot(t_arr, o_forces_sum[:,1], 'g.', markersize=1)
plt.plot(t_arr, o_forces_sum[:,2], 'b.', markersize=1)
plt.hlines(0, t_arr[0]-1, t_arr[-1]+1, 'k')
plt.hlines(-m*g.linear[2], t_arr[0]-1, t_arr[-1]+1, 'k')

plt.figure('IMU GYR')
plt.title('Raw IMU gyro measurements')
plt.plot(t_arr, arr_dic['i_omg_oi'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['i_omg_oi'][:,2], 'b', markersize=1)
plt.xlabel('t (s)')
plt.ylabel('omg (rad/s)')


plt.figure('IMU ACC')
plt.title('Raw IMU acc measurements')
plt.plot(t_arr, arr_dic['imu_acc'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['imu_acc'][:,2], 'b', markersize=1)
plt.xlabel('t (s)')
plt.ylabel('a (m/s^2)')

plt.figure('Mocap translation')
plt.plot(t_arr, arr_dic['w_p_wm'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['w_p_wm'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['w_p_wm'][:,2], 'b', markersize=1)

plt.figure('IMU RPY')
plt.plot(t_arr, rpy_imu_arr[:,0], 'r', markersize=1)
plt.plot(t_arr, rpy_imu_arr[:,1], 'g', markersize=1)
plt.plot(t_arr, rpy_imu_arr[:,2], 'b', markersize=1)

plt.figure('Mocap RPY')
plt.plot(t_arr, rpy_mocap_arr[:,0], 'r', markersize=1)
plt.plot(t_arr, rpy_mocap_arr[:,1], 'g', markersize=1)
plt.plot(t_arr, rpy_mocap_arr[:,2], 'b', markersize=1)

plt.figure('o_a_oi')
plt.plot(t_arr, arr_dic['o_a_oi'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['o_a_oi'][:,2], 'b', markersize=1)

plt.figure('dqa_filt')
plt.plot(t_arr, dqa_filt_arr[:,0], 'r', markersize=1)
plt.plot(t_arr, dqa_filt_arr[:,1], 'g', markersize=1)
plt.plot(t_arr, dqa_filt_arr[:,2], 'b', markersize=1)

plt.figure('qa')
plt.plot(t_arr, arr_dic['qa'][:,0], 'r', markersize=1)
plt.plot(t_arr, arr_dic['qa'][:,1], 'g', markersize=1)
plt.plot(t_arr, arr_dic['qa'][:,2], 'b', markersize=1)


# subsample at 100Hz
tau_arr = arr_dic['tau']

dt = 1e-3
f = 50
# f = 1000
# t_arr_bis = np.arange(N)*dt
A = 0.2
tau_sin_1d = A*np.sin(2*np.pi*f*t_arr) 

# try to remove feedforward
tau_sin = np.tile(tau_sin_1d, 12).reshape(12, N).T
tau_corr_arr = tau_arr - tau_sin

plt.figure('tau')
plt.title('Joint torques (Nm)')
# for i in range(3):
#     plt.plot(t_arr, tau_arr[:,i], 'rgb'[i], markersize=1)
plt.plot(t_arr, tau_arr[:,2], 'rgb'[2], markersize=1)


plt.figure('tau subsampled')
plt.title('Joint torques subsampled (Nm)')
# for phase in range(10):
# for phase in [1, 4, 6, 8, 9]:
for phase in [6]:
    plt.plot(t_arr[phase::10], tau_arr[phase::10,2], markersize=3, label=str(phase))
plt.legend()



plt.figure('o forces for each leg')
NL = 4
for k in range(NL):
    plt.subplot(NL,1,k+1)
    plt.title(LEGS[k])
    # plt.plot(t_arr, o_forces_arr[:,3*k+0], 'r', markersize=1)
    # plt.plot(t_arr, o_forces_arr[:,3*k+1], 'g', markersize=1)
    plt.plot(t_arr, o_forces_arr[:,3*k+2], 'b', markersize=1)
    plt.plot(t_arr, detect_arr[:,k]*THRESH_VIZ, 'k')
    plt.plot(t_arr, contacts[:,k]*(THRESH_VIZ+0.1), 'r')


if SHOW:
    plt.show()