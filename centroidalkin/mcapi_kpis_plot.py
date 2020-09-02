import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kpi_heatmaps(rmse_path, maxe_path):
    df_rmse = pd.read_csv(rmse_path, index_col=0)
    df_maxe = pd.read_csv(maxe_path, index_col=0)

    plt.figure('RMSE')
    plt.title('RMSE')
    sns.heatmap(df_rmse, annot=True, linewidths=.5)

    plt.figure('Max abs error')
    plt.title('Max abs error')
    sns.heatmap(df_maxe, annot=True, linewidths=.5)
    plt.show()


if __name__ == "__main__":
    FIG_DIR_PATH = 'figs/metrics/'
    SUB_DIR = 'test'
    PATH = FIG_DIR_PATH + SUB_DIR + "/"
    rmse_path = PATH+'rmse.csv'
    maxe_path = PATH+'maxe.csv'
    
    plot_kpi_heatmaps(rmse_path, maxe_path)