import numpy as np
from numpy.lib.function_base import append
from numpy.linalg.linalg import inv
import pinocchio as pin
import matplotlib.pyplot as plt
import plotly.offline as pltly

import seaborn as sns
from cov_model import points_projection, compute_covariance, get_cam_model, get_cam_model_cv
from ellipsoid import Ellipsoid, ellipsoid_meshes, plot_ellipsoid_3d_mpl
from plot_utils import plot_frames, set_axes_equal



def plot_projs(points_lst, ax, colors=None):
    # plot projections
    if colors is None:
        colors = len(points_lst)*[None]
    for i, points in enumerate(points_lst):
        plot_tag_projection(points, color=colors[i])
    plt.xlim(0, width)
    plt.ylim(0, height)
    ax.set_aspect('equal')


def plot_tag_projection(points, color=None):
    pts = points.copy()
    # pts[:,1] = -pts[:,1] + height  # not needed if matplotlib ax.invert_yaxis()

    corners_col = 'rgbk'
    # plot only the corners
    for i, p in enumerate(pts):
        plt.plot(p[0], p[1], corners_col[i]+'x')
    # plot point junctions
    for i in range(4):
        if color is None:
            c = corners_col[i]
        else:
            c = color
        if i == 3:
            x12, y12 = [pts[3,0], pts[0,0]], [pts[3,1], pts[0,1]]

            plt.plot(x12, y12, c)
        else:
            x12, y12 = pts[i:i+2,0], pts[i:i+2,1]
            plt.plot(x12, y12, c)


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





def generate_transfo():
    """
    Generate a bunch of transformations representing the 6D
    pose of apriltags in the camera frame.
    """

    T_lst = []
    x_range = -0.15, 0, 0.15
    z_range = np.arange(1, 5)
    for z in z_range:
        for x in x_range:
            R = np.eye(3)
            # R = pin.rpy.rpyToMatrix(np.deg2rad([10, 0, 0]))
            R = pin.rpy.rpyToMatrix(np.deg2rad([45, 0, 0]))
            T = pin.SE3(R, np.array([3*z*x, 0, z]))
            T_lst.append(T)

    # T_lst = []
    # T = pin.SE3(pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0, 0])), np.array([0.0, 0, 0.5])); T_lst.append(T)
    # T = pin.SE3(pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0, 0])), np.array([0.0, 0, 0.6])); T_lst.append(T)
    # T = pin.SE3(pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0, 0])), np.array([0.1, 0, 0.5])); T_lst.append(T)
    # T = pin.SE3(pin.rpy.rpyToMatrix(np.deg2rad([30, 0, 0])), np.array([0.0, 0, 2])); T_lst.append(T)
    # T = pin.SE3(pin.rpy.rpyToMatrix(np.deg2rad([, 0, 0])), np.array([0.0, 0, 2])); T_lst.append(T)

    return T_lst


if __name__ == '__main__':

    width, height, K = get_cam_model('camera_realsense.yml')
    print(K)
    # width, height, K = get_cam_model_cv('camera_mohamed.yml')
    # print(K)
    # tag_width = 0.158  # LAAS_solo_walk_11_21
    tag_width = 0.15
    # tag_width = 0.3
    sig_pix = 2.0
    a_corners = 0.5*tag_width*np.array([
        [-1,  1, 0], # bottom left
        [ 1,  1, 0], # bottom right
        [ 1, -1, 0], # top right
        [-1, -1, 0], # top left
    ])

    T_lst = generate_transfo()
    # express all quantities in a alternate world frame "w prime" -> does not really work
    # wp_T_w = pin.SE3(pin.rpy.rpyToMatrix(0,-np.pi/2,np.pi/2), np.zeros(3))
    # T_lst = [wp_T_w*T for T in T_lst]
    
    points_lst, Q_lst = project_multiple(T_lst, K, a_corners, sig_pix)
    # chi square ctrical value for dimension 3 and 99 confidence 3 sigma interval
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
    chi2_conf = 11.345
    
    # We cannot represent the full 6D covariance of the  
    # Q_lst = [chi2_conf*Q[:3,:3] for Q in Q_lst]  # POSITION COV
    Q_lst = [chi2_conf*Q[3:6,3:6] for Q in Q_lst]  # ORIENTATION COV
    ells = cov2ellipses(T_lst, Q_lst)

    plt.rcParams["figure.figsize"] = (15,10) 
    # MATPLOTLIB
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    plot_frames(ax, pin.SE3.Identity(), la=0.2, ms=10, lw=1)  # camera frame
    # plot_frames(ax, wp_T_w, 0.2, ms=10, lw=2)  # camera frame, world turned (DOES NOT WORK)
    plot_frames(ax, T_lst, la=0.3, ms=10, lw=1)  # tag frames
    for ell in ells:
        plot_ellipsoid_3d_mpl(ax, ell, color='orange')
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
    plt.title('Apriltag projection', fontsize=30)
    ax = fig.add_subplot()
    colors = ['black', 'lightcoral', 'orange']
    # print(points_lst)
    plot_projs(points_lst, ax)
    # plot_projs(points_lst, ax, colors=colors)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.invert_yaxis()
    # plt.grid()
    # plt.savefig('apriltag_proj.pdf', format='pdf')



    # PLOTLY
    # plot_plotly(ells)


    plt.show()
