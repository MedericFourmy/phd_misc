'''
Some variations around the computations of the centroidal momentum.

Conclusion:
use pio.computeCentroidalMap(rmodel,rdata,q) to get the matrix c_Ag s.t.:
c_Ag * vq = c_Ag * [ b_v_b,b_w,qdot_A ] 
          = [ m*0_cd, 0_Lc ]
          = [ m*0_cd, 0_Ic*0_w+Aq*qdot_A ]
'''

import numpy as np
import pinocchio as pio
from example_robot_data import loadTalos
np.set_printoptions(precision=1,threshold=1e4,suppress=True,linewidth=200)
from numpy.linalg import norm,svd,eig,inv,pinv

r = loadTalos()
rm = r.model
rd = rm.createData()

q0 = r.q0.copy()
q = pio.randomConfiguration(rm)
vq = np.random.rand(rm.nv)*2-1

b_Mc = pio.crba(rm,rd,q) # mass matrix at b frame expressed in b frame
b_I_b = b_Mc[:6,:6]      # inertia matrix at b frame expressed in b frame
Jc = pio.jacobianCenterOfMass(rm,rd,q)

w_T_b = rd.oMi[1]         # world to body transform

# Check that the Inertia matrix expressed at the center of mass is of the form:
# c_I_c = b_Mc[:6,:6] = [[m*Id3  03 ]
#                        [03    I(q)]]
pio.centerOfMass(rm, rd, q, vq)
c_T_b = pio.SE3(w_T_b.rotation, -rd.com[0] + w_T_b.translation)
# Ic = cXb^star * Ib * bXc
c_I_c = c_T_b.actionInverse.T @ b_I_b @ c_T_b.actionInverse  # momentum coordinate transform
assert(np.allclose(c_I_c[:3,:3], rd.mass[0] * np.eye(3)))

# Check that M[:6,:] is the centroidal momentum expressed in F_b
c_hc  = pio.computeCentroidalMomentum(rm,rd,q,vq)
b_hc  = pio.Force(b_Mc[:6,:] @ vq)
assert( (c_hc - c_T_b * b_hc ).isZero())

# Check that the linear momentum is m*cd
pio.centerOfMass(rm,rd,q,vq)
cd = rd.vcom[0]
m = rd.mass[0]
np.allclose(c_hc.linear, m*cd)      

# Transforms vq to "centroid vel configuration vect" in universe frame
Z = np.vstack([
    Jc,
    np.hstack([np.zeros([3,3]),w_T_b.rotation,np.zeros([3,rm.nv-6])]),
    np.hstack([np.zeros([rm.nv-6,6]),np.eye(rm.nv-6)])
    ])

b_w = vq[3:6]
o_w = w_T_b.rotation @ b_w
cd__w__q = np.concatenate([cd, o_w, vq[6:]])
assert(np.allclose(cd__w__q, Z @ vq))

# Check that Ac = [ m*I_3 0_3  0_nqA; 
#                   0_3   Ic   Aq    ] 
# and c_hc = Ac*[cd,o_w,qdot_A]
Ac = c_T_b.actionInverse.T @ b_Mc[:6,:] @ inv(Z)
assert(np.allclose(Ac[:3,:3] / m,np.eye(3)))  # Ac[:3,3:] = m I_3
assert(norm(Ac[:3,3:])<1e-6)
assert(norm(Ac[3:,:3])<1e-6)
o_Ic = Ac[3:6,3:6]
assert(np.allclose(o_Ic,o_Ic.T))
assert(np.all(eig(o_Ic)[0]>0))

# Check the centroidal map is M[:6,:] in the F_c basis
c_X_b_star = c_T_b.actionInverse.T
Ag = pio.computeCentroidalMap(rm,rd,q)
assert(np.allclose(c_X_b_star @ b_Mc[:6,:], Ag))
assert(np.allclose(Ac @ Z, Ag))

# "gesticulation" angular momentum
#q_static = q.copy()
#q_static[:7] = 6*[0] + [1]
#vq_static = vq.copy()
#vq_static[:7] = 6*[0]
#pio.centerOfMass(rm, rd, q_static, vq_static)

