import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from covariance_model import *



width, height, K = get_cam_model('camera_realsense.yml')
tag_width = 0.1

a_corners = 0.5*tag_width*np.array([
    [-1,  1, 0], # bottom left
    [ 1,  1, 0], # bottom right
    [ 1, -1, 0], # top right
    [-1, -1, 0], # top left
])


T = pin.SE3.Identity()
# T = pin.SE3.Random()
t, R = T.translation, T.rotation
t[0] = 0.15
t[2] = 0.5
# R = pin.rpy.rpyToMatrix(np.deg2rad([0.0, 0.0, 45]))
R = pin.rpy.rpyToMatrix(np.deg2rad([0.0, 60, 0]))

points = points_projection(t, R, K, a_corners)
# transform from camera pixel frame to matplotlib frame
points[:,1] = -points[:,1] + height

print(points)

# plotting apriltag point
fig = plt.figure()
c = 'rgbk'
# plot only the corners
for i, p in enumerate(points):
    plt.plot(p[0], p[1], c[i]+'x')
# plot point junctions
for i in range(4):
    if i == 3:
        x12, y12 = [points[3,0], points[0,0]], [points[3,1], points[0,1]]
        print(x12)
        plt.plot(x12, y12, c[i])
    else:
        x12, y12 = points[i:i+2,0], points[i:i+2,1]
        print(x12)
        plt.plot(x12, y12, c[i])


plt.xlim(0, width)
plt.ylim(0, height)
fig.axes[0].set_aspect('equal')

plt.show()
