import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_naming import dirname_from_params


def plot_kpi_heatmaps(rmse_path, maxe_path, scale_dist, mass_dist):
    df_rmse = pd.read_csv(rmse_path, index_col=0)
    df_maxe = pd.read_csv(maxe_path, index_col=0)

    plt.figure('RMSE')
    plt.title('RMSE\n scale_dist: {}, mass_dist: {}'.format(scale_dist, mass_dist))
    sns.heatmap(df_rmse, annot=True, linewidths=.5)

    plt.figure('Max abs error')
    plt.title('Max abs error\n scale_dist: {}, mass_dist: {}'.format(scale_dist, mass_dist))
    sns.heatmap(df_maxe, annot=True, linewidths=.5)
    plt.show()


if __name__ == "__main__":
    SCALE_DIST = 0.01
    MASS_DIST = False
    # SCALE_DIST = 0.1
    # MASS_DIST = True

    # Select plots to activate
    RUN_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/bin/mcapi_povcdl_estimation'
    PARAM_FILE = '/home/mfourmy/Documents/Phd_LAAS/wolf/bodydynamics/demos/mcapi_povcdl_estimation.yaml'

    FIG_DIR_PATH = 'figs/metrics/'

    with open(PARAM_FILE, 'r') as fr:
        params = yaml.safe_load(fr)
    params['scale_dist']    = SCALE_DIST
    params['mass_dist']     = MASS_DIST

    SUB_DIR = dirname_from_params(params)
    PATH = FIG_DIR_PATH + SUB_DIR + "/"

    rmse_path = PATH+'rmse.csv'
    maxe_path = PATH+'maxe.csv'
    
    plot_kpi_heatmaps(rmse_path, maxe_path, SCALE_DIST, MASS_DIST)