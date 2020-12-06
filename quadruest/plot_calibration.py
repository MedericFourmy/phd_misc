import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# file_x = 'x_est.csv'
# file_rmse = 'rmse_res.csv'
file_x = 'x_est_manual_longer_analytic_13x5000.csv'
file_rmse = 'rmse_res_manual_longer_analytic_13x5000.csv'
# file_x = 'x_est_manual_longer_numeric_13x5000.csv'
# file_rmse = 'rmse_res_manual_longer_numeric_13x5000.csv'


x_arr = pd.read_csv(file_x).to_numpy()
rmse_res_arr = pd.read_csv(file_rmse).to_numpy()

qa_delta_arr = x_arr[:,:12]
dab_arr = x_arr[:,12:]

dab_arr_gtr = np.array([
    0.327,
    0.522,
    0.405,
    0.403,
    0.523,
    0.331
])
LEGS = ['FL', 'FR', 'HR', 'HL']

combinations = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3),
]
comb_names = ['{}/{}'.format(LEGS[c[0]], LEGS[c[1]]) for c in combinations]


joints = np.array([[leg+'1', leg+'2', leg+'3'] for leg in LEGS]).flatten()

plt.figure('deltas')
plt.title('deltas estimated')
plt.xlabel('articulation')
plt.ylabel('angle (deg)')
plt.plot(joints, np.rad2deg(qa_delta_arr).T, 'x')

plt.figure('distances')
plt.title('distances estimated')
plt.xlabel('Feet pairs')
plt.ylabel('distance (mm)')
plt.plot(comb_names, dab_arr.T*1000, 'x')
plt.plot(comb_names, dab_arr_gtr*1000, 'o', label='ruller gtr')

plt.figure('rmse')
plt.title('rmse residuals')
plt.xlabel('Feet pairs')
plt.ylabel('distance (mm)')
plt.plot(comb_names, rmse_res_arr.T*1000, 'x')









plt.show()
