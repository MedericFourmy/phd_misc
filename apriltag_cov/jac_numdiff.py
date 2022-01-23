import numpy as np
from apriltag_cov.cov_model import *

EPS = 1e-9

def compute_num_jac(f, x, rplus=np.add):
    """
    Compute numerical jacobian wrt. to each element of x.
    """
    y = f(x)
    nx, ny = len(x), len(y)
    J = np.zeros((ny, nx))

    eps_mat = EPS*np.eye(nx)

    for i in range(nx):
        x_pert = rplus(x, eps_mat[i,:])
        J[:,i] = (f(x_pert) - f(x))/EPS
    
    return J


def rplus_so3(R, omg):
    return R @ pin.exp3(omg)


def dummy_f(x):
    """
    From wiki: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
    """
    x1, x2 = x
    return np.array([x1**2*x2, 5*x1 + np.sin(x2)])

def dummy_jac(x):
    x1, x2 = x
    return np.array([        
        [2*x1*x2, x1**2],
        [5, np.cos(x2)],
    ])


if __name__ == '__main__':
    # first verify numerical diff function itself

    x = np.array([0.4, 0.6])
    J_x = dummy_jac(x)
    J_x_num = compute_num_jac(dummy_f, x)

    
    print('J_x - J_x_num')
    print(J_x - J_x_num)

    # Now our optical projection functions
    h = np.random.random(3)
    J_u_h = h2pix_jac(h)
    J_u_h_num = compute_num_jac(h2pix, h)
    
    print()
    print('J_u_h - J_u_h_num')
    print(J_u_h - J_u_h_num)

    # For pinhole and points_projection, we need to diferentiate partially with respect to t and R
    # on top of that R belongs to SO3 so we need to account for that
    import pinocchio as pin

    T = pin.SE3.Identity()
    t, R = T.translation, T.rotation
    t[2] = 1

    tag_width = 0.1
    a_corners = 0.5*tag_width*np.array([
        [-1,  1, 0], # bottom left
        [ 1,  1, 0], # bottom right
        [ 1, -1, 0], # top right
        [-1, -1, 0], # top left
    ])
    # K = np.array([
    #     [30,0, 5],
    #     [0, 20,4],
    #     [0, 0, 1],
    # ])
    K = np.eye(3)

    # Pinhole function
    corner = a_corners[0,:]
    pinhole_t = lambda t: pinhole(t, R, K, corner)
    pinhole_R = lambda R: pinhole(t, R, K, corner)

    J_pinhole_tR = pinhole_jac(t, R, K, corner)
    J_pinhole_t, J_pinhole_R = J_pinhole_tR[:,:3], J_pinhole_tR[:,3:6]
    J_pinhole_t_num = compute_num_jac(pinhole_t, t)
    J_pinhole_R_num = compute_num_jac(pinhole_R, R, rplus=rplus_so3)

    print()
    print('J_pinhole_t - J_pinhole_t_num')
    print(J_pinhole_t - J_pinhole_t_num)

    print()
    print('J_pinhole_R - J_pinhole_R_num')
    print(J_pinhole_R - J_pinhole_R_num)


    points_projection_t = lambda t: points_projection(t, R, K, a_corners)
    points_projection_R = lambda R: points_projection(t, R, K, a_corners)

    J_points_projection_tR = points_projection_jac(t, R, K, a_corners)
    J_points_projection_t, J_points_projection_R = J_points_projection_tR[:,:3], J_points_projection_tR[:,3:6]
    J_points_projection_t_num = compute_num_jac(points_projection_t, t)
    J_points_projection_R_num = compute_num_jac(points_projection_R, R, rplus=rplus_so3)

    print()
    print('J_points_projection_t - J_points_projection_t_num')
    print(J_points_projection_t - J_points_projection_t_num)

    print()
    print('J_points_projection_R - J_points_projection_R_num')
    print(J_points_projection_R - J_points_projection_R_num)




    