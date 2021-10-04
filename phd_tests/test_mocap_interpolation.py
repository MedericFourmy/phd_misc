import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

t_arr = np.array([0,1,2])
q_arr = np.array([[0,0,0,1], 
                  [0, 0, 0.7071068, 0.7071068], 
                  [0,0,1,0]])

# upsample rotations 
r_arr = R.from_quat(q_arr)
slerp = Slerp(t_arr, r_arr)

t_interp = np.linspace(0,2,11)
r_interp = slerp(t_interp)

print(r_arr.as_euler('xyz', degrees=True))
print(r_interp.as_euler('xyz', degrees=True))
print(r_interp.as_quat())


print('\nHandling nans')
q_arr[1] = np.array([np.nan]*4)

idx_nnan = np.where(~np.isnan(q_arr[:,0]))[0]
q_arr_nnan = q_arr[idx_nnan]
t_arr_nnan = t_arr[idx_nnan]

r_arr_nnan = R.from_quat(q_arr_nnan)
slerp = Slerp(t_arr_nnan, r_arr_nnan)
r_interp = slerp(t_arr)

print(r_arr.as_euler('xyz', degrees=True))
print(r_interp.as_euler('xyz', degrees=True))
print(r_interp.as_quat())


# translations
p_arr = np.linspace(np.zeros(3), np.ones(3), 11)
p_arr[3] = np.array([np.nan]*3)

p_arr_interp = pd.DataFrame(p_arr).interpolate().to_numpy()


