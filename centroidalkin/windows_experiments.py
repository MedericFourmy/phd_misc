#!/usr/bin/env python
# coding: utf-8

import os
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import subprocess
import itertools
import matplotlib.pyplot as plt
import pinocchio as pin
from experiment_naming import dirname_from_params

PLOT = False

def nb_possibilities(lst):
    param_nb_lst = [len(l) for l in lst]
    return np.prod(param_nb_lst)

def rmse(err_arr):
    return np.sqrt(np.mean(err_arr**2))

def create_bak_file_and_get_params(path):
    path_bak = path+'_bak'
    shutil.copyfile(path, path_bak)
    with open(path, 'r') as fr:
        params = yaml.safe_load(fr)
    return params

def restore_initial_file(path):
    path_bak = path+'_bak'
    shutil.move(path_bak, path)

def diff_shift(arr):
    diff = np.roll(arr, -1) - arr
    # last element will be huge since roll wraps around the array
    # -> just copy the n_last - 1
    diff[-1] = diff[-2]
    return diff


# Select plots to activate
RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/solo_real_pov_estimation'
PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/solo_real_estimation.yaml'
PROC_IMU_PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/processor_imu_solo12.yaml'
TREE_PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/params_tree_manager.yaml'

FIG_DIR_PATH = 'figs/window_experiments/'

res = input('Deleting previous results (y)?')
if res == 'y':
    shutil.rmtree(FIG_DIR_PATH, ignore_errors=True)

params_imu = create_bak_file_and_get_params(PROC_IMU_PARAM_FILE)
params = create_bak_file_and_get_params(PARAM_FILE)
params_tree = create_bak_file_and_get_params(TREE_PARAM_FILE)

if not os.path.exists(FIG_DIR_PATH):
    os.makedirs(FIG_DIR_PATH)

max_t_kf_lst = [0.01, 0.05, 0.3, 1]
KF_nb_lst = [10, 30, 50, 100, 200]
fz_thresh_lst = [2]

# max_t_kf_lst = [0.05]
# KF_nb_lst = [5, 100]
# fz_thresh_lst = [1]

possibs = nb_possibilities([max_t_kf_lst, KF_nb_lst, fz_thresh_lst])
print('Combinations to evaluate: ', possibs)

RESULTS = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments_results/out.npz'

for i, (max_t_kf, KF_nb, fz_thresh) in enumerate(itertools.product(max_t_kf_lst, KF_nb_lst, fz_thresh_lst)):

    params['fz_thresh'] = fz_thresh
    with open(PARAM_FILE, 'w') as fw: yaml.dump(params, fw)

    params_imu['keyframe_vote']['max_time_span'] = max_t_kf
    with open(PROC_IMU_PARAM_FILE, 'w') as fw: yaml.dump(params_imu, fw)

    params_tree['config']['problem']['tree_manager']['n_frames'] = KF_nb
    with open(TREE_PARAM_FILE, 'w') as fw: yaml.dump(params_tree, fw)

    # run executable
    t1 = time.time()
    subprocess.run(RUN_FILE, stdout=subprocess.DEVNULL)
    print('time: ', time.time()-t1)

    
    config = {
        'max_t_kf': max_t_kf,
        'KF_nb': KF_nb,
        'fz_thresh': fz_thresh
    }
    with open(FIG_DIR_PATH+'conf_{}.yaml'.format(i), 'w') as fw: yaml.dump(config, fw)

    # plot and save the results
    arr_dic = np.load(RESULTS)
    fig = plt.figure()
    plt.title('Position')
    plt.plot(arr_dic['t'], arr_dic['o_p_ob'][:,0], 'r')
    plt.plot(arr_dic['t'], arr_dic['o_p_ob'][:,1], 'g')
    plt.plot(arr_dic['t'], arr_dic['o_p_ob'][:,2], 'b')
    plt.savefig(FIG_DIR_PATH+'pos_{}.png'.format(i))
    plt.close(fig=fig)

    arr_dic = np.load(RESULTS)
    fig = plt.figure()
    plt.title('Position Diff')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_p_ob'][:,0]), 'r')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_p_ob'][:,1]), 'g')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_p_ob'][:,2]), 'b')
    plt.savefig(FIG_DIR_PATH+'diff_pos_{}.png'.format(i))
    plt.close(fig=fig)
    
    fig = plt.figure()
    plt.title('Velocity')
    plt.plot(arr_dic['t'], arr_dic['o_v_ob'][:,0], 'r')
    plt.plot(arr_dic['t'], arr_dic['o_v_ob'][:,1], 'g')
    plt.plot(arr_dic['t'], arr_dic['o_v_ob'][:,2], 'b')
    plt.savefig(FIG_DIR_PATH+'vel_{}.png'.format(i))
    plt.close(fig=fig)

    arr_dic = np.load(RESULTS)
    fig = plt.figure()
    plt.title('Velocity Diff')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_v_ob'][:,0]), 'r')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_v_ob'][:,1]), 'g')
    plt.plot(arr_dic['t'], diff_shift(arr_dic['o_v_ob'][:,2]), 'b')
    plt.savefig(FIG_DIR_PATH+'diff_vel_{}.png'.format(i))
    plt.close(fig=fig)


 
restore_initial_file(PARAM_FILE)
restore_initial_file(PROC_IMU_PARAM_FILE)
restore_initial_file(TREE_PARAM_FILE)


