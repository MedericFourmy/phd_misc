import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

N = 1000
sig = 0.1
t = np.linspace(0, 10, N)
y = np.sin(t)
z = y + np.random.normal(0, sig, N)

plt.figure()
# plt.plot(t, y, 'g', label='y')
# plt.plot(t, z, 'b', label='z')
# filtered traj
plt.plot(t, gaussian_filter1d(z, 2), 'c', alpha=0.5, label='2')
plt.plot(t, gaussian_filter1d(z, 4), 'k', alpha=0.5, label='4')
plt.plot(t, gaussian_filter1d(z, 6), 'y', alpha=0.5, label='6')
plt.legend()
plt.show()