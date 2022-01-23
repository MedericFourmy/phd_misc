from collections.abc import Iterable
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
Example taken from Barfoot "State Estimation", 4.1.1 Full Bayesian Estimation

Goal: examine various approximations of the posterior distribution
p(x|y) = p(y|x)*p(x)/p(y)

where p(y|x) is a nonlinear measurement model
p(x) ~ N(0, sig) is a gaussian prior

p(x|y) is NOT a gaussian posterior
"""

# measurement model example: 1D stereo vision
"""
"This is the type of nonlinearity present in a stereo camera (cf., Fig-
ure 4.1), where the state, x, is the position of a landmark (in metres),
the measurement, y, is the disparity between the horizontal coordinates
of the landmark in the left and right images (in pixels), f is the focal
length (in pixels), b is the baseline (horizontal distance between left and
right cameras; in metres), and n is the measurement noise (in pixels)."

x = landmark depth
y = disparity observed by the cam
f = focal length (pixels)
b = baseline of the cameras (m)
y = u - v = fb/x
"""

f = 400  # pixels
b = 0.1  # m

# true location
x_true = 22

# prior distribution of x
x_prior = 20
sigx = 2

# measurement additive noise
sigy = 3e-1


def stereo(x):
    return f*b/x

def stereo_jac(x):
    return -f*b/x**2

def stereo_n(x):
    if not isinstance(x, Iterable):
        l = 1
    else:
        l = len(x)
    return stereo(x) + np.random.normal(0, sigy, size=l)


################
# Grid Inference
################


def describe(x_grid, dens):
    "Describe a grid based density probability function"

    dx = get_dx(x_grid)

    dens_max = np.max(dens)
    imax = np.argmax(dens)
    dens_mode = x_grid[imax]
    dens_mean = sum(x_grid*dens*dx)
    dens_variance = sum((dens_mean - x_grid)**2*dens**dx)

    return dens_mode, dens_max, dens_mean, dens_variance


def compute_post(x_grid, prior, mmodel, y):
    """
    Grid based 
    """
    dx = get_dx(x_grid)
    likelihood = stats.norm.pdf(mmodel(x_grid), y, sigy)
    post_un = prior*likelihood
    return post_un / sum(post_un*dx)


def get_dx(x_grid):
    return (np.roll(x_grid, -1) - x_grid)[1]


################
# MAP optimization
################
def compute_map(x_init, y_lst, Nmax=10):
    """
    Computes MAP and information matrix of the posterior.

    MAP from vanilla Gauss Newton algo
    L(x) =  0.5 ||r||^2
    L(x + dx) = L(x) + 0.5*(2*r^T @ J @ dx  +  dx^T @ J^T @ J @ dx)
              = L(x) + r^T @ J @ dx  + 0.5*dx^T @ J^T @ J @ dx
    argmin_dx L(x + dx) ?  -> dL/dx = 0  -> J^T @ J @ dx = -J^T @ r
    
    Hessian of L with respect to x:
    H = J^T @ J 

    Since L is the -log likelihood of the posterior distribution, 
    Laplacian approximation theorem states that H is a the best 
    approximation of the information matrix of the state posterior.
    """

    x = np.array([x_init])
    for _ in range(Nmax):
        J = jac_res(x, y_lst)
        r = res(x, y_lst)
        dx = np.linalg.solve(J.T@J, -J.T@r)  # Gauss Newton step
        dx = dx.reshape(x.shape)
        x = x + dx

        if abs(dx) < 1e-6:
            break
    
    H = J.T@J  # Laplace approximation
    return x, H


def res(x, y_lst):
    """
    Whitened/scaled residuals with convention:
    r = h(x) - y
    """
    prior_res = (x - x_prior)/sigx
    hx = stereo(x)
    meas_res_lst = [(hx - y)/sigy for y in y_lst]
    return np.concatenate([prior_res]+meas_res_lst).reshape((len(y_lst)+1, 1))


def jac_res(x, y_lst):
    """
    Jacobian of the whitened residuals.
    """

    j_prior = np.ones(1)/sigx
    jh = stereo_jac(x)/sigy
    j_meas = [jh]*len(y_lst)
    return np.concatenate([j_prior]+j_meas).reshape((len(y_lst)+1, 1))





if __name__ == '__main__':

    # Create noisy measurements
    Nmeas = 10
    y_arr = stereo_n(x_true*np.ones(Nmeas))

    #####################################
    # Grid approximation of the posterior
    #####################################
    Ng = 10000  # number of grid elements
    window = 15
    x_grid = np.linspace(x_prior-window, x_prior+window, Ng)
    prior = stats.norm.pdf(x_grid, x_prior, sigx)

    # Recursive bayesian inferance
    post = prior  # just for initialization
    # series of measurement -> recursive bayesian inference
    post_lst = []
    for i in range(Nmeas):
        post = compute_post(x_grid, post, stereo, y_arr[i])
        post_lst.append(post)

    post_mode, post_max, post_mean, post_variance = describe(x_grid, post_lst[-1]) 
    print('post_mode: ', post_mode)

    ####################################
    # MAP approximation of the posterior
    ####################################
    x_init = x_prior
    x_map, H = compute_map(x_init, y_arr)
    Q = 1/H  # np.linalg.inv(H) # covariance
    sig_map = np.sqrt(Q)


    #######
    # PLOTS
    #######
    post_lst = post_lst[-1:]  # only last posterior

    plt.figure()
    plt.plot(x_grid, prior, 'k--', label='prior')
    alphas = reversed(np.linspace(1, 0, len(post_lst))**2)
    for p, a in zip(post_lst, alphas):
        plt.plot(x_grid, p, 'k', alpha=a, label='Grid posterior')
    plt.vlines(post_mode, 0, 1.2*post_max, 'red', linestyles='--', label='posterior mode')
    plt.vlines(post_mean, 0, 1.2*post_max, 'green', linestyles='--', label='posterior mean')

    # map distrib
    post_map = stats.norm.pdf(x_grid, x_map, sig_map).flatten()
    plt.plot(x_grid, post_map, 'r', alpha=1, label='MAP posterior')
    plt.xlabel('distance x (m)')
    plt.ylabel('Probability densities')
    plt.xlim(15,25)
    plt.legend()

    # plt.savefig('MAP_stereo1D.pdf', format='pdf')

    plt.show()
