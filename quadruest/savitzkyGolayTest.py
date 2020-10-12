# Savitzky-Golay filtering giving incorrect derivative in 1D
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

N = 1000
dt = 1e-2
std = 0.005
t = dt*np.arange(N)
x = np.sin(t)
dxdt = np.cos(t)
y = x + std*np.random.normal(0,1,N)

# filters
# finite diff
y_prev = np.roll(y, 1)
y_prev[0] = y_prev[1]
diff = (y - y_prev)/dt

Zn = signal.savgol_filter(y, window_length=5, polyorder=3, deriv=0, mode='mirror')
Zf = signal.savgol_filter(y, window_length=15, polyorder=5, deriv=1, delta=dt, mode='mirror')


plt.figure()
plt.subplot(2,1,1)
plt.plot(t,x, label = 'Input')
plt.plot(t,y, label = 'Input noisy')
plt.plot(t,Zn, label= 'Savitzky-Golay filtered')
plt.subplot(2,1,2)
plt.plot(t,dxdt, label = 'Input derivative')
plt.plot(t,diff, label = 'Finite derivative')
plt.plot(t,Zf, label= 'Savitzky-Golay filtered - 1st derivative')
plt.legend()
plt.show()