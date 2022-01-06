import numpy as np
from numpy.linalg.linalg import inv
import pinocchio as pin
import matplotlib.pyplot as plt
import plotly.offline as pltly

import seaborn as sns
from covariance_model import points_projection, compute_covariance, get_cam_model
from ellipsoid import Ellipsoid, ellipsoid_meshes, plot_ellipsoid_3d_mpl, set_axes_equal


def plot_tag_projection(points):
    pts = points.copy()
    pts[:,1] = -pts[:,1] + height

    c = 'rgbk'
    # plot only the corners
    for i, p in enumerate(pts):
        plt.plot(p[0], p[1], c[i]+'x')
    # plot point junctions
    for i in range(4):
        if i == 3:
            x12, y12 = [pts[3,0], pts[0,0]], [pts[3,1], pts[0,1]]
            plt.plot(x12, y12, c[i])
        else:
            x12, y12 = pts[i:i+2,0], pts[i:i+2,1]
            plt.plot(x12, y12, c[i])


def project_multiple(T_lst, K, a_corners, sig_pix):
    points_lst = [points_projection( T.translation, T.rotation, K, a_corners).reshape((4,2)) for T in T_lst]
    Q_lst = [compute_covariance(T.translation, T.rotation, K, a_corners, sig_pix) for T in T_lst]

    return points_lst, Q_lst


def cov2ellipses(T_lst, Q_lst):
    ells = []
    for i in range(len(T_lst)):
        # API takes the information matrix, NOT the covariance
        A = np.linalg.inv(Q_lst[i].copy())
        ell = Ellipsoid(T_lst[i].translation, A)
        ells.append(ell)
    return ells


def plot_plotly(ells):
    # plotly plot covariances
    meshes = [ellipsoid_meshes(ell) for ell in ells]
    pltly.plot(meshes)


def plot_mpl(ells, ax):
    plot_ellipsoid_3d_mpl(ells[0], ax)
    plot_ellipsoid_3d_mpl(ells[1], ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)


def plot_frames(T_lst, ax):
    pass


def plot_projs(points_lst, ax):
    # plot projections
    for points in points_lst:
        plot_tag_projection(points)
    plt.xlim(0, width)
    plt.ylim(0, height)
    fig.axes[0].set_aspect('equal')



if __name__ == '__main__':

    width, height, K = get_cam_model('camera_realsense.yml')
    tag_width = 0.2
    sig_pix = 3.0
    a_corners = 0.5*tag_width*np.array([
        [-1,  1, 0], # bottom left
        [ 1,  1, 0], # bottom right
        [ 1, -1, 0], # top right
        [-1, -1, 0], # top left
    ])


    t1 = np.array([0.15, 0, 2.0])
    R1 = pin.rpy.rpyToMatrix(np.deg2rad([0.0, 80, 0]))
    T1 = pin.SE3(R1, t1)

    t2 = np.array([-0.15, 0, 2.0])
    R2 = pin.rpy.rpyToMatrix(np.deg2rad([0.0, -80, 0]))
    T2 = pin.SE3(R2, t2)

    T_lst = [T1, T2]

    points_lst, Q_lst = project_multiple(T_lst, K, a_corners, sig_pix)
    Q_lst = [Q[:3,:3] for Q in Q_lst]
    ells = cov2ellipses(T_lst, Q_lst)

    # PLOTLY
    plot_plotly(ells)

    plt.rcParams["figure.figsize"] = (15,10) 
    # MATPLOTLIB
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    plot_mpl(ells, ax)

    # 2D projection of tags
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_projs(points_lst, ax)


    plt.show()
