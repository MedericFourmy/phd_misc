import numpy as np
import pandas as pd
import pinocchio as pin


# Remarks
# - MPI data files time synchronized but not all of the same length

# Reference frames
# - b: base frame
# - w: world inertial frame of mocap
# - m: mocap markers frame -> will later consider b = m for orientation and velocity
# - o: universe inertial frame reference of the imu
# - i: imu


# data available:
# t
# imu_acc
# o_a_oi
# o_q_i
# i_omg_oi
# qa
# dqa
# tau
# w_pose_wm
# m_v_wm
# w_v_wm
# o_R_i
# w_p_wm
# w_q_m
# w_R_m

def read_data_files_mpi(folder, dt, delimiter=None):
    """
    folder: folder path (ending with /), leading to the dat files
    dt: signals frequencies
    delimiter: .dat files delimiters, None has to be used if default sot format used (accounts for \t, space(s))
    """

    # IMU
    imu_acc_file = folder+'dg_solo12-imu_accelerometer.dat'    # direct imu reading (+g on z)
    o_a_oi_file = folder+'dg_solo12-imu_linear_acceleration.dat'  # compensated for gravity -> in inertial frame?
    o_q_i_file = folder+'dg_solo12-imu_attitude_quaternion.dat'
    i_omg_oi_file = folder+'dg_solo12-imu_gyroscope.dat'
    # Encoders
    qa_file = folder+'dg_solo12-joint_positions.dat'
    dqa_file = folder+'dg_solo12-joint_velocities.dat'
    # torques
    tau_file = folder+'dg_solo12-joint_torques.dat'
    # Vicon
    w_pose_wm_file = folder+'dg_solo12_vicon_client-solo12_position.dat'
    m_v_wm_file = folder+'dg_solo12_vicon_client-solo12_velocity_body.dat'
    w_v_wm_file = folder+'dg_solo12_vicon_client-solo12_velocity_world.dat'

    # imu_acc: raw imu measurement
    # o_a_oi: litteraly the imu ref frame origin acceleration (gravity is compensated!)
    # TODO: m_v_wm is in fact a 6D vel -> split
    # TODO: w_v_wm is 6D spatial vector or lin vel is classical one?
    file_dic = {
        'imu_acc': imu_acc_file,
        'o_a_oi': o_a_oi_file,
        'o_q_i': o_q_i_file, 
        'i_omg_oi': i_omg_oi_file, 
        'qa': qa_file, 
        'dqa': dqa_file, 
        'tau': tau_file, 
        'w_pose_wm': w_pose_wm_file, 
        'm_v_wm': m_v_wm_file, 
        'w_v_wm': w_v_wm_file, 
    }
    
    # Hyp (verified): all files start with same timestamp -> plainly remove first column
    arr_dic = {key: np.loadtxt(file_path, delimiter=delimiter)[:,1:] for key, file_path in file_dic.items()}
    N = min(arr.shape[0] for key, arr in arr_dic.items())
    # remove all data after the minimum signal length
    arr_dic = {key: arr[:N,:] for key, arr in arr_dic.items()}

    # shortest time array fitting all traj
    t_arr = np.arange(N)*dt  # in seconds, starting from 0
    arr_dic['t'] = t_arr

    # other preprocessing (to unify formats)
    # beware of quaternion conventions! it seems that this dataset is storing quaternions vals 
    # in qw,qx,qy,qz order while usually (Eigen, pinocchio) it is 
    # in qx,qy,qz,qw
    # -> roll the entire quaternion trajectory columns
    arr_dic['o_q_i'] = np.roll(arr_dic['o_q_i'],-1,axis=1) 
    arr_dic['o_R_i'] = np.array([pin.Quaternion(o_q_i.reshape(4,1)).toRotationMatrix() for o_q_i in arr_dic['o_q_i']])  # if q: qx,qy,qz,qw
    arr_dic['w_p_wm'] = arr_dic['w_pose_wm'][:,:3]
    arr_dic['w_q_m'] = arr_dic['w_pose_wm'][:,3:]
    # TODO: likely to also use same qw,qx,qy,qz convention (to be tested)
    # arr_dic['w_q_m'] = np.roll(arr_dic['w_q_m'],-1,axis=1) 
    arr_dic['w_R_m'] = np.array([pin.Quaternion(w_q_m.reshape(4,1)).toRotationMatrix() for w_q_m in arr_dic['w_pose_wm'][:,3:]])
    # roll/pitch/yaw
    arr_dic['o_rpy_i'] = np.array([pin.rpy.matrixToRpy(R) for R in arr_dic['o_R_i']])
    arr_dic['w_rpy_m'] = np.array([pin.rpy.matrixToRpy(R) for R in arr_dic['w_R_m']])

    return arr_dic


def read_data_file_laas(file_path, dt):
    # Load data file
    data = np.load(file_path)

    # i_a_oi: litteraly the imu ref frame origin acceleration (gravity is compensated), expressed in base frame
    arr_dic = {
        'imu_acc': data['baseAccelerometer'],
        'i_a_oi': data['baseLinearAcceleration'],
        'o_q_i': data['baseOrientation'], 
        'i_omg_oi': data['baseAngularVelocity'], 
        'qa': data['q_mes'], 
        'dqa': data['v_mes'], 
        'tau': data['torquesFromCurrentMeasurment'],     
        'w_p_wm': data['mocapPosition'], 
        'w_q_m': data['mocapOrientationQuat'], 
        'w_R_m': data['mocapOrientationMat9'], 
        'w_v_wm': data['mocapVelocity']
    }

    N = min(arr.shape[0] for key, arr in arr_dic.items())
    arr_dic = {key: arr[:N,:] for key, arr in arr_dic.items()}
    # shortest time array fitting all traj
    t_arr = np.arange(N)*dt  # in seconds, starting from 0
    arr_dic['t'] = t_arr

    # other preprocessing (to unify formats)
    arr_dic['w_pose_wm'] = np.hstack([arr_dic['w_p_wm'], arr_dic['w_q_m']])
    arr_dic['m_v_wm'] = np.array([w_R_m.T @ w_v_wm for w_R_m, w_v_wm in zip(arr_dic['w_R_m'], arr_dic['w_v_wm'])])   # compute local mocap base velocity
    arr_dic['o_R_i'] = np.array([pin.Quaternion(o_q_i.reshape(4,1)).toRotationMatrix() for o_q_i in arr_dic['o_q_i']])
    # roll/pitch/yaw
    arr_dic['o_rpy_i'] = np.array([pin.rpy.matrixToRpy(R) for R in arr_dic['o_R_i']])
    arr_dic['w_rpy_m'] = np.array([pin.rpy.matrixToRpy(R) for R in arr_dic['w_R_m']])

    # acceleration in world frame
    arr_dic['o_a_oi'] = np.array([o_R_i@i_a_oi for o_R_i, i_a_oi in zip(arr_dic['o_R_i'], arr_dic['i_a_oi'])]) 

    if 'contactStatus' in data:
        arr_dic['contactStatus'] = data['contactStatus']

    return arr_dic

def read_data_traj_dat(traj_path, dt):
    # !!!!!!!!!!
    # HYP: IMU supposed to be centered at b -> i = m = b
    # !!!!!!!!!!
    q_path = traj_path+'_q.dat'
    v_path = traj_path+'_v.dat'
    tau_path = traj_path+'_tau.dat'

    df_q = pd.read_csv(q_path, sep=' ', index_col=0, header=None)
    df_v = pd.read_csv(v_path, sep=' ', index_col=0, header=None)
    df_tau = pd.read_csv(tau_path, sep=' ', index_col=0, header=None)

    arr_dic = {}
    t_arr = np.arange(df_q.shape[0])*dt  # in seconds, starting from 0
    arr_dic['t'] = t_arr
    arr_dic['qa'] = df_q.iloc[:,7:].to_numpy()
    arr_dic['w_pose_wm'] = df_q.iloc[:,:7].to_numpy()
    arr_dic['w_p_wm'] = arr_dic['w_pose_wm'][:,:3]
    arr_dic['w_q_m'] = arr_dic['w_pose_wm'][:,3:]
    arr_dic['w_R_m'] = np.array([pin.Quaternion(q).toRotationMatrix() for q in arr_dic['w_q_m']])

    arr_dic['dqa'] = df_v.iloc[:,6:].to_numpy()
    arr_dic['m_v_wm'] = df_v.iloc[:,:3].to_numpy()
    arr_dic['i_omg_oi'] = df_v.iloc[:,3:6].to_numpy()
    arr_dic['w_v_wm'] = np.array([w_R_m@m_v_wm for w_R_m, m_v_wm in zip(arr_dic['w_R_m'], arr_dic['m_v_wm'])])

    arr_dic['tau'] = df_tau.iloc[:].to_numpy()

    # dummy
    arr_dic['o_R_i'] = arr_dic['w_R_m']
    arr_dic['o_q_i'] = arr_dic['w_q_m']

    # things we don't have: acceleration in imu frame, acc_meas...
    # Obtain through windowed finite difference and then remove messy data at beginning and end
    Nd = 1
    w_p_wm_prev_arr = np.roll(arr_dic['w_p_wm'],  Nd, axis=0)
    w_p_wm_post_arr = np.roll(arr_dic['w_p_wm'], -Nd, axis=0)
    arr_dic['o_a_oi'] = np.array([  (w_p_wm0 + w_p_wm2 - 2*w_p_wm1)/((Nd*dt)**2)      
                            for w_p_wm0, w_p_wm1, w_p_wm2 in zip(w_p_wm_prev_arr, arr_dic['w_p_wm'], w_p_wm_post_arr)])

    arr_dic['i_a_oi'] = np.array([o_R_i.T@o_a_oi for o_R_i, o_a_oi in zip(arr_dic['o_R_i'], arr_dic['o_a_oi'])])
    G = 9.806
    w_g = np.array([0,0,-G])
    arr_dic['imu_acc'] = np.array([i_a_oi - w_R_m.T@w_g for w_R_m, i_a_oi in zip(arr_dic['w_R_m'], arr_dic['i_a_oi'])])


    return shortened_arr_dic(arr_dic, Nd, -Nd)


def shortened_arr_dic(arr_dic, S, N=None):
    if N is None:
        N = len(arr_dic['t'])
    return {k: arr_dic[k][S:N] for k in arr_dic}


if __name__ == '__main__':
    import time
    from example_robot_data import load

    SLEEP = True

    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_09.npz'  # Standing still (5s), mocap 200Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_10.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_16.npz'  # Moving up (5s), mocap 200Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_17.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_25.npz'  # Standing still (5s), mocap 500Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_26.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_29.npz'  # Moving up (5s), mocap 200Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_30.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_31.npz'  # Moving up->front->down (10s), mocap 500Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_32.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_54.npz'  # Moving up then random movements (15s), mocap 500Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_56.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_57.npz'  # //
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_15_59.npz'  # Already in air, random movements (15s), mocap 500Hz
    # file_name = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Experiments_MocapIMU_2021_04_23/data_2021_04_23_16_03.npz'

    file_name = '/home/mfourmy/Downloads/data_2021_05_19_16_34.npz'

    # dt = 1e-3
    dt = 2*1e-3
    arr_dic = read_data_file_laas(file_name, dt)  # if default format
    t_arr = arr_dic['t']
    w_pose_wm_arr = arr_dic['w_pose_wm']
    qa_arr = arr_dic['qa']

    robot = load('solo12')
    print(robot.nv)
    robot.initViewer()
    robot.initViewer(loadModel=True)

    gui = robot.viewer.gui
    CAMERA_TRANSFORM = [7.660114765167236,
                        3.531687021255493,
                        0.9310312271118164,
                        0.39686328172683716,
                        0.5807958245277405,
                        0.5837690830230713,
                        0.40544962882995605]
    gui.setCameraTransform('python-pinocchio', CAMERA_TRANSFORM)
    gui.addFloor('world/floor')
    gui.setLightingMode('world/floor', 'OFF')

    print('GO!')
    for i, t in enumerate(t_arr):
        t1 = time.time()
        w_pose_wm = w_pose_wm_arr[i,:]
        qa = qa_arr[i,:]

        q = np.hstack([w_pose_wm, qa])
        robot.display(q)

        delay = time.time() - t1
        if SLEEP and delay < dt:
            time.sleep(dt - delay)

    print('STOP!')

