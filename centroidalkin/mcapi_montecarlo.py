
import os
import time
import subprocess
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from experiment_naming import dirname_from_params_path

# number of experiments
N = 20
nb_sig = 3
df_kfs_lst = []
df_cov_lst = []
df_gtr_kfs_lst = []

SAVE_FIGURES = True
FIG_DIR_PATH = 'figs/MC/'
SUB_DIR = dirname_from_params_path('/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/mcapi_povcdl_estimation.yaml')
PATH = FIG_DIR_PATH + SUB_DIR + "/"
if not os.path.exists(PATH):
    os.makedirs(PATH)


for i in range(N):
    t1 = time.time()
    subprocess.run('/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_povcdl_estimation', stdout=subprocess.DEVNULL)
    print('Run {}/{}, time: {}'.format(i, N, time.time()-t1))
    df_kfs = pd.read_csv('kfs.csv')
    df_cov = pd.read_csv('cov.csv')
    df_gtr = pd.read_csv('gtr.csv')

    # filter the ground truth trajectory to recover values at keyframe times
    df_gtr_kfs = pd.DataFrame()
    for t in df_kfs['t']: 
        df_gtr_kfs = df_gtr_kfs.append(df_gtr.loc[np.isclose(df_gtr['t'], t)], ignore_index=True)

    df_kfs_lst.append(df_kfs)
    df_cov_lst.append(df_cov)
    df_gtr_kfs_lst.append(df_gtr_kfs)


# TODO: orientation
names = ['P', 'V', 'C', 'D', 'L', 'B']
prefixes = ['p', 'v', 'c', 'cd', 'L', 'bp']

for name, pre in zip(names, prefixes):
    fig = plt.figure(name+' errors')
    plt.title(name+' estimates+cov')
    legend_printed = False
    for i in range(N):
        plt.scatter(df_kfs_lst[i]['t'], df_kfs_lst[i][pre+'x'] - df_gtr_kfs_lst[i][pre+'x'], c='r', s=3, label=pre+'x')
        plt.scatter(df_kfs_lst[i]['t'], df_kfs_lst[i][pre+'y'] - df_gtr_kfs_lst[i][pre+'y'], c='g', s=3, label=pre+'y')
        plt.scatter(df_kfs_lst[i]['t'], df_kfs_lst[i][pre+'z'] - df_gtr_kfs_lst[i][pre+'z'], c='b', s=3, label=pre+'z')

        if not legend_printed:
            plt.legend()
            legend_printed = True
    
    # Not necessary to print uncertainty bounds for all runs (almost all the sames), just recover the last one
    sig_xx = np.sqrt(df_cov_lst[i]['Q'+pre+'x'])
    sig_yy = np.sqrt(df_cov_lst[i]['Q'+pre+'y'])
    sig_zz = np.sqrt(df_cov_lst[i]['Q'+pre+'z'])
    plt.plot(df_cov_lst[i]['t'], -nb_sig*sig_xx, alpha=0.3, c='r')
    plt.plot(df_cov_lst[i]['t'],  nb_sig*sig_xx, alpha=0.3, c='r')
    plt.plot(df_cov_lst[i]['t'], -nb_sig*sig_yy, alpha=0.3, c='g')
    plt.plot(df_cov_lst[i]['t'],  nb_sig*sig_yy, alpha=0.3, c='g')
    plt.plot(df_cov_lst[i]['t'], -nb_sig*sig_zz, alpha=0.3, c='b')
    plt.plot(df_cov_lst[i]['t'],  nb_sig*sig_zz, alpha=0.3, c='b')
    file_path = PATH+name+'.png' 
    fig.savefig(file_path)
    print('Saved '+file_path)


# if SAVE_FIGURES:
#     file_path = PATH+'biases_errs_sigmas.png' 
#     fig.savefig(file_path)
#     print('Saved '+file_path)

plt.show()