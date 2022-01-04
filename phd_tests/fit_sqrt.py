import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
t = np.arange(500)*dt
alpha = 1.5
y = alpha*np.sqrt(t) + np.random.normal(0,0.05,t.shape[0])

def fit_sqrt(t, y):
    return sum(y*np.sqrt(t))/sum(t)


alpha_fit = fit_sqrt(t, y)
y_fit = alpha_fit*np.sqrt(t)
print(alpha_fit)

plt.figure()
plt.plot(t, y, 'b.', label='data')
plt.plot(t, y_fit, 'g', label='fitted')
plt.xlabel('t (s)')
plt.legend()
plt.show()