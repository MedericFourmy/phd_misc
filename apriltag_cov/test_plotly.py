import numpy as np
import plotly.offline as plt
import plotly.graph_objects as go
from numpy import sin, cos, pi


NUM = 50

def ellipse(pos, abc):
    phi = np.linspace(0, 2*pi, num=NUM)
    theta = np.linspace(-pi/2, pi/2, NUM)
    phi, theta=np.meshgrid(phi, theta)

    x = pos[0] + abc[0]*cos(theta) * sin(phi)
    y = pos[1] + abc[1]*cos(theta) * cos(phi)
    z = pos[2] + abc[2]*sin(theta)

    print(x.shape)

    return x.flatten(), y.flatten(), z.flatten()


def create_mesh(x, y, z):
    # The alphahull parameter sets the shape of the mesh. 
    # If the value is -1 (default value) then Delaunay triangulation is used. 
    # If >0 then the alpha-shape algorithm is used. 
    # If 0, the convex hull is represented (resulting in a convex body).
    return go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     alphahull=0.0, 
                     flatshading=True,
                     color='lightpink', 
                     opacity=0.50,
                     name='Ellipse',
                     showscale=False
                     )


pos = np.zeros(3)
abc = np.ones(3)
abc[0] += 0.5

x1, y1, z1 = ellipse(pos, abc)
pos[0] += 1
x2, y2, z2 = ellipse(pos, abc)


mesh1 = create_mesh(x1, y1, z1) 
mesh2 = create_mesh(x2, y2, z2) 


plt.plot([mesh1])



### might need to wait for plot to download before copying
# time.sleep(1)

# copyfile('{}/{}.svg'.format(dload, img_name),
#          '{}/{}.svg'.format(save_dir, img_name))