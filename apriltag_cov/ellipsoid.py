import math
import numpy as np
import plotly.graph_objects as go

from plot_utils import set_axes_equal


# Copy pasta of some part of https://github.com/kbarbary/nestle/blob/master/nestle.py

# -----------------------------------------------------------------------------
# Helpers

def vol_prefactor(n):
    """Volume constant for an n-dimensional sphere:
    for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
    for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
    """
    if n % 2 == 0:
        f = 1.
        i = 2
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2
    else:
        f = 2.
        i = 3
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2

    return f

def randsphere(n, rstate=np.random):
    """Draw a random point within an n-dimensional unit sphere"""

    z = rstate.randn(n)
    return z * rstate.rand()**(1./n) / np.sqrt(np.sum(z**2))



# -----------------------------------------------------------------------------
# Ellipsoid

class Ellipsoid:
    """An N-ellipsoid.
    Defined by::
        (x - v)^T A (x - v) = 1
    where the vector ``v`` is the center of the ellipse and ``A`` is an N x N
    matrix. Assumes that ``A`` is symmetric positive definite.
    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape ``(N,)``
        Coordinates of ellipse center. Note that the array is *not* copied.
        This array is never modified internally.
    A : `~numpy.ndarray` with shape ``(N, N)``
        Matrix describing the axes. Watch out! This array is *not* copied.
        but may be modified internally!
    """

    def __init__(self, ctr, a):
        self.n = len(ctr)
        self.ctr = ctr    # center coordinates
        self.a = a        # ~ inverse of covariance of points contained
        self.vol = vol_prefactor(self.n) / np.sqrt(np.linalg.det(a))

        # eigenvalues (l) are a^-2, b^-2, ... (lengths of principle axes)
        # eigenvectors (v) are normalized principle axes
        l, v = np.linalg.eigh(a)
        self.axlens = 1. / np.sqrt(l)

        # Scaled eigenvectors are the axes: axes[:,i] is the i-th
        # axis.  Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere into a point in the ellipsoid.
        self.axes = np.dot(v, np.diag(self.axlens))

    def scale_to_vol(self, vol):
        """Scale ellipoid to satisfy a target volume."""
        f = (vol / self.vol) ** (1.0 / self.n)  # linear factor
        self.a *= f**-2
        self.axlens *= f
        self.axes *= f
        self.vol = vol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis"""
        i = np.argmax(self.axlens)  # which is the major axis?
        v = self.axes[:, i]  # vector to end of major axis
        return self.ctr - v, self.ctr + v

    def contains(self, x):
        """Does the ellipse contain the point?"""
        d = x - self.ctr
        return np.dot(np.dot(d, self.a), d) <= 1.0

    def randoffset(self, rstate=np.random):
        """Return an offset from ellipsoid center, randomly distributed
        within ellipsoid."""
        return np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def sample(self, rstate=np.random):
        """Chose a sample randomly distributed within the ellipsoid.
        Returns
        -------
        x : 1-d array
            A single point within the ellipsoid.
        """
        return self.ctr + self.randoffset(rstate=rstate)

    def samples(self, nsamples, rstate=np.random):
        """Chose a sample randomly distributed within the ellipsoid.
        Returns
        -------
        x : (nsamples, ndim) array
            Coordinates within the ellipsoid.
        """

        x = np.empty((nsamples, self.n), dtype=np.float)
        for i in range(nsamples):
            x[i, :] = self.sample(rstate=rstate)
        return x

    def __repr__(self):
        return "Ellipsoid(ctr={})".format(self.ctr)


def plot_ellipsoid_3d_mpl(ax, ell, color='#2980b9'):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.1)



def ellipsoid_meshes(ell):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    NUM = 20

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, NUM)
    v = np.linspace(0.0, np.pi, NUM)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])
    print('x.max()')
    print(x.min())
    print(x.max())


    # The alphahull parameter sets the shape of the mesh. 
    # If the value is -1 (default value) then Delaunay triangulation is used. 
    # If >0 then the alpha-shape algorithm is used. 
    # If 0, the convex hull is represented (resulting in a convex body).
    return go.Mesh3d(x=x.flatten(),
                     y=y.flatten(),
                     z=z.flatten(),
                     alphahull=0.0, 
                     flatshading=True,
                     color='lightpink', 
                     opacity=0.50,
                     name='Ellipse',
                     showscale=False
                     )




if __name__ == '__main__':
    import pinocchio as pin

    ctr1 = np.array([0,0,0])
    ctr2 = np.array([2,0,0])
    # define INFORMATION matrix
    A = 0.25*np.array([
        [1, 0, 0,],
        [0, 1, 0,],
        [0, 0, 1,],
    ])
    R = pin.rpy.rpyToMatrix(np.deg2rad([0, 45, 0]))
    Q = R.T @ A @ R
    ell1 = Ellipsoid(ctr1, A)
    ell2 = Ellipsoid(ctr2, Q)

    # MATPLOTLIB == CRAP for 3D
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (15,10) 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    plot_ellipsoid_3d_mpl(ax, ell1)
    plot_ellipsoid_3d_mpl(ax, ell2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()

    # PLOTLY
    # import plotly.offline as plt
    # mesh1 = ellipsoid_meshes(ell1)
    # mesh2 = ellipsoid_meshes(ell2)
    # plt.plot([mesh1, mesh2])


    print(ell1.axes)