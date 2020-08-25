import numpy as np
import pinocchio as pin
from example_robot_data import loadTalos


SCALE = 0.01
TXT_FILE = 'talos_dist{}.txt'.format(SCALE)
READ_FROM_TXT = False
PLOT_DISPARITY = False

# retrieve the normal Talos model
r = loadTalos()
rmodel = r.model
rdata = r.data
 

# create a brand new model object (discard the data)
rmodel_dist = loadTalos().model
if READ_FROM_TXT:
    rmodel_dist.loadFromText(TXT_FILE)
else:
    for iner in rmodel_dist.inertias: 
        iner.lever += SCALE*(np.random.rand(3) - 0.5)

rdata_dist = rmodel_dist.createData() 

# CoM
c = pin.centerOfMass(rmodel, rdata, r.q0)
c_dist = pin.centerOfMass(rmodel_dist, rdata_dist, r.q0)
print('c_dist - c')
print(c_dist - c)

# Mass matrix
M = pin.crba(rmodel, rdata, r.q0)
M_dist = pin.crba(rmodel_dist, rdata_dist, r.q0)

print('M_dist - M')
print(M_dist[:6, :6] - M[:6, :6])

# save the result as a txt file:
rmodel_dist.saveToText(TXT_FILE)


if PLOT_DISPARITY:
    # funny plots

    # number of configuration to sample
    N = 5000
    biases = np.zeros((N,3))
    for i in range(N):
        q = pin.randomConfiguration(rmodel)
        # Base at the origin
        q[:6] = 0
        q[6] = 1
        p_bc = pin.centerOfMass(rmodel, rdata, q)
        p_bc_dist = pin.centerOfMass(rmodel_dist, rdata_dist, q)
        biases[i,:] = p_bc_dist - p_bc

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, Y, Z = biases[:,0], biases[:,1], biases[:,2]

    # 2D figures
    plt.figure('y=f(x)')
    plt.scatter(X, Y, 'rx')
    plt.figure('z=f(y)')
    plt.scatter(Y, Z, 'rx')
    plt.figure('x=f(z)')
    plt.scatter(Z, X, 'rx')


    # 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = biases[:,0], biases[:,1], biases[:,2]
    ax.scatter(X, Y, Z, c='r', marker='^', s=1)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    plt.show()