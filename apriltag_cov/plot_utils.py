import numpy as np
import pinocchio as pin
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_frames(ax, T_lst, la, ms=20, lw=3):
    """
    ax: axes elt
    T_lst: list (or single) SE3 frame
    la: length of the frame axes (m)
    ms: mutation_scale (arrow end)
    lw: line width/thickness
    """
    if not isinstance(T_lst, list):
        T_lst = [T_lst]
    if not isinstance(la, list):
        la = len(T_lst)*[la]
    print(la)
    for T, la in zip(T_lst, la):
        _plot_frame(ax, T, la, ms, lw)


def _plot_frame(ax, T, la, ms=20, lw=3):
    """
    ax: axes elt
    T_lst: list (or single) of worl_T_local SE3 frame
    la: length of the frame axes (m)
    ms: mutation_scale (arrow end)
    lw: line width/thickness    
    """

    t, R = T.translation, T.rotation
    for i in range(3):
        v = t + la*R[:,i]
        plt.plot(t[0], t[1], t[2], 'o', markersize=3, color='k', alpha=0.5)
        # mutation scale = scale of the arrow end
        a = Arrow3D([t[0], v[0]], [t[1], v[1]], [t[2], v[2]], 
                    mutation_scale=ms, 
                    lw=lw, arrowstyle="-|>", color='rgb'[i])
        ax.add_artist(a)


def set_axes_equal(ax, limits=None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if limits is None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    else:
        x_limits, y_limits, z_limits = limits

    print('x_limits', x_limits)
    print('y_limits', y_limits)
    print('z_limits', z_limits)

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    


if __name__ == '__main__':
    T_lst = []
    T = pin.SE3.Identity(); T_lst.append(T)
    T = pin.SE3.Identity(); T.translation[0] = 0.4; T_lst.append(T)
    T = pin.SE3.Identity(); T.rotation = pin.rpy.rpyToMatrix(0,0,np.pi/4); T_lst.append(T)
    T = pin.SE3.Identity(); T.translation[0] = 0.4; T.rotation = pin.rpy.rpyToMatrix(0,0,np.pi/4); T_lst.append(T)


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plot_frames(ax, T_lst, 0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # # in degrees
    # ax.azim = -50
    # ax.elev = -50

    plt.title('Frames')

    # plt.draw()  # not necesarry but was there in the example
    plt.show()