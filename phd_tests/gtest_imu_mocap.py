import numpy as np
import pandas as pd
import pinocchio as pin
import matplotlib.pyplot as plt

df_gtr = pd.read_csv('gtest_gtr.csv')
df_int = pd.read_csv('gtest_int.csv')
df_est = pd.read_csv('gtest_est.csv')
df_bias = pd.read_csv('gtest_bias.csv')
t = df_gtr.t


plt.figure()
plt.title('Position ')
for i in range(3):
    l = 'xyz'[i]
    c = 'rgb'[i]
#     plt.subplot(3,1,i+1)
    plt.plot(t, df_gtr['p'+l], c+'-' , label='gtr')
    plt.plot(t, df_int['p'+l], c+'--', label='int')
    plt.plot(t, df_est['p'+l], c+'.' , label='est')
    plt.legend()

plt.figure()
plt.title('Velocity ')
for i in range(3):
    l = 'xyz'[i]
    c = 'rgb'[i]
    # plt.subplot(3,1,i+1)
    plt.plot(t, df_gtr['v'+l], c+'-' , label='gtr')
    plt.plot(t, df_int['v'+l], c+'--', label='int')
    plt.plot(t, df_est['v'+l], c+'.' , label='est')
    plt.legend()

# orientation log
qcols = ['qx', 'qy', 'qz', 'qw']
q_arr_gtr = df_gtr[qcols].to_numpy()
q_arr_int = df_int[qcols].to_numpy()
q_arr_est = df_est[qcols].to_numpy()

o_arr_gtr = np.array([pin.log(pin.Quaternion(q.reshape((4,1))).toRotationMatrix()) for q in q_arr_gtr])
o_arr_int = np.array([pin.log(pin.Quaternion(q.reshape((4,1))).toRotationMatrix()) for q in q_arr_int])
o_arr_est = np.array([pin.log(pin.Quaternion(q.reshape((4,1))).toRotationMatrix()) for q in q_arr_est])

plt.figure()
plt.title('Orientation ')
for i in range(3):
    c = 'rgb'[i]
    # plt.subplot(3,1,i+1)
    plt.plot(t, o_arr_gtr[:,i], c+'-' , label='gtr')
    plt.plot(t, o_arr_int[:,i], c+'--', label='int')
    plt.plot(t, o_arr_est[:,i], c+'.' , label='est')
    plt.legend()


plt.figure()
plt.title('Imu biases')
plt.subplot(2,1,1)
plt.plot(df_bias.t, df_bias['bax'], 'r.')
plt.plot(df_bias.t, df_bias['bay'], 'g.')
plt.plot(df_bias.t, df_bias['baz'], 'b.')
plt.subplot(2,1,2)
plt.plot(df_bias.t, df_bias['bwx'], 'r.')
plt.plot(df_bias.t, df_bias['bwy'], 'g.')
plt.plot(df_bias.t, df_bias['bwz'], 'b.')

plt.show()



