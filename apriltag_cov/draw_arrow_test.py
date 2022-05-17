import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

####################################################
# This part is just for reference if
# you are interested where the data is
# coming from
# The plot is at the bottom
#####################################################

N = 300

# Generate some example data
mu_vec = np.array([0,0,0])
cov_mat = 0.1*np.array([[1,0,0],[0,1,0],[0,0,1]])
samples = np.random.multivariate_normal(mu_vec, cov_mat, N)

# mean values
mean_x = mean(samples[:,0])
mean_y = mean(samples[:,1])
mean_z = mean(samples[:,2])

#eigenvectors and eigenvalues
eig_val, eig_vec = np.linalg.eig(cov_mat)

################################
#plotting eigenvectors
################################    

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.plot(samples[:,0], samples[:,1], samples[:,2], '.', markersize=5, color='g', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec:
    #ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
    #I will replace this line with:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], 
                [mean_z, v[2]], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvectors')

plt.draw()
plt.show()