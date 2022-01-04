#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

nb_trials = 200

N = 10000
dt = 1e-3
Dt = N*dt
t_arr = dt*np.arange(N)
sig_v = 0.01

p_arr = np.zeros((N, nb_trials))
for j in range(nb_trials):
    p = 0
    for i in range(N):
        p_arr[i,j] = p
        noise_v = np.random.normal(0, sig_v)
        p += noise_v*dt


def noise_envelop(t, dt, sig_v, nb_sig):
    return nb_sig*np.sqrt(t*dt)*sig_v

env_arr = noise_envelop(t_arr, dt, sig_v, 3)

plt.figure('noise integration with 3sigma bounds')
plt.plot(t_arr, p_arr, '.', markersize=0.2)
plt.plot(t_arr,  env_arr, 'black')
plt.plot(t_arr, -env_arr, 'black')

plt.figure()
plt.hist(p_arr[-1,:], 20)

plt.show()
