import numpy as np
from numpy.linalg.linalg import inv
import pinocchio as pin
import matplotlib.pyplot as plt
import plotly.offline as pltly

import seaborn as sns
from covariance_model import points_projection, compute_covariance, get_cam_model
from ellipsoid import Ellipsoid, ellipsoid_meshes, plot_ellipsoid_3d_mpl
from plot_utils import plot_frames, set_axes_equal


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




def plot_projs(points_lst, ax):
    # plot projections
    for points in points_lst:
        plot_tag_projection(points)
    plt.xlim(0, width)
    plt.ylim(0, height)
    fig.axes[0].set_aspect('equal')


def generate_transfo():
    x_range = -0.15, 0, 0.15
    z_range = np.arange(1, 5)

    T_lst = []

    for z in z_range:
        for x in x_range:
            R = np.eye(3)
            # R = pin.rpy.rpyToMatrix(np.pi/10, 0, 0)
            T = pin.SE3(R, np.array([3*z*x, 0, z]))
            T_lst.append(T)

    return T_lst

    # t1 = np.array([0.15, 0, 2.0])
    # R1 = pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0, 0]))
    # T1 = pin.SE3(R1, t1)

    # t2 = np.array([-0.15, 0, 2.0])
    # R2 = pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0, 0]))
    # T2 = pin.SE3(R2, t2)


if __name__ == '__main__':

    width, height, K = get_cam_model('camera_realsense.yml')
    tag_width = 0.2
    sig_pix = 2.0
    a_corners = 0.5*tag_width*np.array([
        [-1,  1, 0], # bottom left
        [ 1,  1, 0], # bottom right
        [ 1, -1, 0], # top right
        [-1, -1, 0], # top left
    ])

    T_lst = generate_transfo()
    # express all quantities in a alternate world frame "w prime"
    # wp_T_w = pin.SE3(pin.rpy.rpyToMatrix(0,-np.pi/2,np.pi/2), np.zeros(3))
    # T_lst = [wp_T_w*T for T in T_lst]
    
    points_lst, Q_lst = project_multiple(T_lst, K, a_corners, sig_pix)
    # chi square ctrical value for dimension 3 and 99 confidence 3 sigma interval
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
    chi2_conf = 11.345
    # chi2_conf = 500
    Q_lst = [chi2_conf*Q[:3,:3] for Q in Q_lst]  # POSITION ERROR
    # Q_lst = [chi2_conf*Q[3:6,3:6] for Q in Q_lst]  # ORIENTATION ERROR
    ells = cov2ellipses(T_lst, Q_lst)

    plt.rcParams["figure.figsize"] = (15,10) 
    # MATPLOTLIB
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    plot_frames(ax, pin.SE3.Identity(), la=0.3, ms=10, lw=2)  # camera frame
    # plot_frames(ax, wp_T_w, 0.2, ms=10, lw=2)  # camera frame
    plot_frames(ax, T_lst, la=0.4, ms=10, lw=1)  # tag frames
    for ell in ells:
        plot_ellipsoid_3d_mpl(ax, ell, color='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # in degrees
    # ax.azim = -50
    # ax.elev = -50
    # limits = (
    #     (-0.2, 0.5),
    #     (-0.2, 0.5),
    #     (-0.2, 1.5),
    # )
    # set_axes_equal(ax, limits)
    set_axes_equal(ax)

    # 2D projection of tags
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_projs(points_lst, ax)



    # PLOTLY
    # plot_plotly(ells)


    plt.show()
