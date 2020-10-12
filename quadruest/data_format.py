import numpy as np
from data_readers import read_data_file_laas, read_data_files_mpi, shortened_arr_dic

DATA_FOLDER = '/home/mfourmy/Documents/Phd_LAAS/data/quadruped_experiments/'
IN_FILE_NAME = 'data_2020_10_08_10_04_StandingStill.npz'
OUT_FILE_NAME = 'data_2020_10_08_10_04_StandingStill_shortened_format.npz'

dt = 1e-3  # discretization timespan
# Load LAAS solo8 data file
# arr_dic = read_data_file_laas('quadruped_experiments/data.npz', dt)
arr_dic = read_data_file_laas(DATA_FOLDER+IN_FILE_NAME, dt)
arr_dic = shortened_arr_dic(arr_dic, 2000)

# Load MPI solo12 data files
# folder = "data/solo12_standing_still_2020-10-01_14-22-52/2020-10-01_14-22-52/"
# folder = "data/solo12_com_oscillation_2020-10-01_14-22-13/2020-10-01_14-22-13/"
# folder = "data/solo12_stamping_2020-09-29_18-04-37/2020-09-29_18-04-37/"
# arr_dic = read_data_files_mpi(folder, dt, delimiter=',')  # with "," delimiters

np.savez(DATA_FOLDER+OUT_FILE_NAME, **arr_dic)
print(DATA_FOLDER+OUT_FILE_NAME, ' saved')