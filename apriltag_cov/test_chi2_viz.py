
import numpy as np
import matplotlib.pyplot as plt
from ellipsoid import Ellipsoid, plot_ellipsoid_3d_mpl, set_axes_equal

# chi square confidence, 3D - 99% confidence
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
chi2_conf = 11.345

N = 30000

# Generate some example data
mu_vec = np.array([0,0,0])
cov_mat = 0.1*np.array([[2,0,0],[0,1,0],[0,0,1.5]])
samples = np.random.multivariate_normal(mu_vec, cov_mat, N)

A = np.linalg.inv(chi2_conf*cov_mat)
ell = Ellipsoid(mu_vec, A)
# 99% of the sampled points should be in the ellipsoid -> and it is!
perc_in = sum(ell.contains(pt) for pt in samples)/N
print('Percentage of samples inside the confidence ellipse:')
print(perc_in)

fig = plt.figure()
plt.title('Chi2 test 99% confidence ellipse')
ax = fig.add_subplot(projection='3d') 
ax.plot(samples[:,0], samples[:,1], samples[:,2], '.', markersize=3, color='g', alpha=0.1)
plot_ellipsoid_3d_mpl(ax, ell)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)
plt.show()