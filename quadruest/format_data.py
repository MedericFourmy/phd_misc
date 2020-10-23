import numpy as np
from data_readers import read_data_file_laas, read_data_files_mpi, shortened_arr_dic

# DATA_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_09_10_20_soir/'
DATA_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/Logs_15_10_2020/'
IN_FILE_NAME = 'data_2020_10_15_14_34.npz'
OUT_FILE_NAME = 'data_2020_10_15_14_34_format_shortened_CST_Ogyr.npz'

dt = 1e-3  # discretization timespan
# Load LAAS solo8 data file
# arr_dic = read_data_file_laas('quadruped_experiments/data.npz', dt)
arr_dic = read_data_file_laas(DATA_FOLDER+IN_FILE_NAME, dt)
N = len(arr_dic['t'])
arr_dic = shortened_arr_dic(arr_dic, 2000, N=N-200)
# arr_dic = shortened_arr_dic(arr_dic, 5000, 7150)

arr_dic['imu_acc'][:,:] = [-1.38603431e-03, -7.49042047e-04,  9.79471817e+00]
# arr_dic['i_omg_oi'][:,:] = [ 7.94876453e-06,  4.37204760e-03, -2.82067587e-04]

# arr_dic['imu_acc'][:,:] = [-1.38603431e-03, -7.49042047e-04,  9.79471817e+00]
arr_dic['i_omg_oi'][:,:] = [0,0,0]

# Load MPI solo12 data files
# folder = "data/solo12_standing_still_2020-10-01_14-22-52/2020-10-01_14-22-52/"
# folder = "data/solo12_com_oscillation_2020-10-01_14-22-13/2020-10-01_14-22-13/"
# folder = "data/solo12_stamping_2020-09-29_18-04-37/2020-09-29_18-04-37/"
# arr_dic = read_data_files_mpi(folder, dt, delimiter=',')  # with "," delimiters

np.savez(DATA_FOLDER+OUT_FILE_NAME, **arr_dic)
print(DATA_FOLDER+OUT_FILE_NAME, ' saved')