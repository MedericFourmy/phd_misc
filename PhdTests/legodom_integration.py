import numpy as np
from scipy import signal
import pinocchio as pin
import matplotlib.pyplot as plt


def base_vel_from_stable_contact(robot, q, dq, i_omg_oi, o_R_i, cid):
    """
    Assumes forwardKinematics has been called on the robot object with current q dq
    And that q = [0,0,0, 0,0,0,1, qa]
            dq = [0,0,0, 0,0,0, dqa]
    """
    b_T_l = robot.framePlacement(q, cid, update_kinematics=False)
    b_p_bl = b_T_l.translation
    bRl = b_T_l.rotation

    l_v_bl = robot.frameVelocity(q, dq, cid, update_kinematics=False).linear
    # measurement: velocity in world frame
    b_v_ob = - bRl @ l_v_bl + np.cross(b_p_bl, i_omg_oi)
    return o_R_i @ b_v_ob



path = '/opt/openrobots/share/example-robot-data/robots/solo_description'
urdf = path + '/robots/solo12.urdf'

robot = pin.RobotWrapper.BuildFromURDF(urdf, path, pin.JointModelFreeFlyer())
model = robot.model
data = robot.data

LEGS = ['FL', 'FR', 'HL', 'HR']
# LEGS = ['FL', 'FR', 'HR'] # !!! remove HL because of faulty leg in Logs_09_10_20_soir dataset 
nb_feet = len(LEGS)
contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
contact_frame_ids = [robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]

dt = 1e-3
# data_file_path = "/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_34_format_shortened_CST.npz"
data_file_path = "/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_34_format_shortened_CST_0gyr.npz"
# data_file_path = "/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/data_2020_10_15_14_34_format_shortened_CST_Ogyr_Gacc_0dqa.npz"


arr_dic = np.load(data_file_path)
t_arr = arr_dic['t']
N = len(t_arr)

i_omg_oi_arr = arr_dic['i_omg_oi']
qa_arr = arr_dic['qa']
# dqa_arr = arr_dic['dqa']
# dqa_arr_mean = np.mean(dqa_arr, axis=0)
# # dqa_arr -= dqa_arr_mean
# NW = 100
# filterr = signal.gaussian(NW, 50)
# filterr /= np.sum(filterr)
# dqa_filt_arr = np.array([np.convolve(x, filterr) for x in dqa_arr.T]).T

dqa_filt_arr = signal.savgol_filter(qa_arr, window_length=21, polyorder=2, deriv=1, delta=dt, mode='mirror', axis=0)
dqa_arr = dqa_filt_arr


oRi = np.eye(3)

dic_vb = {cid: [] for cid in contact_frame_ids}
dic_pint_lst = {cid: [] for cid in contact_frame_ids}
dic_pint = {cid: np.zeros(3) for cid in contact_frame_ids}
for i in range(N):

    i_omg_oi = i_omg_oi_arr[i,:]
    qa = qa_arr[i,:]
    # dqa = dqa_arr[i,:]
    dqa = dqa_filt_arr[i,:]

    q = np.concatenate([[0,0,0, 0,0,0,1], qa])
    dq = np.concatenate([[0,0,0, 0,0,0], dqa])

    robot.forwardKinematics(q, dq)

    for cid in contact_frame_ids:
        vb_cid = base_vel_from_stable_contact(robot, q, dq, i_omg_oi, oRi, cid)
        dic_vb[cid].append(vb_cid)
        dic_pint_lst[cid].append(dic_pint[cid])
        dic_pint[cid] = dic_pint[cid] + vb_cid*dt


for cid in contact_frame_ids:
    dic_vb[cid] = np.array(dic_vb[cid])
    dic_pint_lst[cid] = np.array(dic_pint_lst[cid])

for cid, cname in zip(contact_frame_ids, contact_frame_names):
    plt.figure('vb frame '+cname)
    plt.plot(t_arr, dic_vb[cid])

    plt.figure('pint frame '+cname)
    plt.plot(t_arr, dic_pint_lst[cid])

    print(cname, np.mean(dic_vb[cid], axis=0))


max_dqa = 0.1

plt.figure('qa')
plt.plot(t_arr, qa_arr[:,:2])
# plt.axis([2, 10, -max_dqa, max_dqa])

plt.figure('dqa')
plt.plot(t_arr, dqa_arr[:,:2])
plt.axis([2, 10, -max_dqa, max_dqa])

plt.figure('dqa filt')
# plt.plot(t_arr, dqa_filt_arr[:-NW+1,:2])
plt.plot(t_arr, dqa_filt_arr)
plt.axis([2, 10, -max_dqa, max_dqa])


plt.show()