import example_robot_data as robex
import pinocchio as pin
import numpy as np
from numpy.linalg import norm,inv

r=robex.loadSolo(False)
rm=r.model
rd=rm.createData()

nqa = rm.nv-6
qa = np.random.rand(nqa)*2-1
qda = np.random.rand(nqa)*2-1
qdda = np.random.rand(nqa)*2-1

w = np.random.rand(3)*2-1
o_a = np.random.rand(3)*2-1
wdot = np.random.rand(3)*2-1
oR1 = pin.SE3.Random().rotation
taua = np.random.rand(nqa)*2-1



#q = pin.randomConfiguration(rm)
#v = np.random.rand(rm.nv)*2-1
#a = np.random.rand(rm.nv)*2-1

q = np.concatenate([np.zeros(3),pin.Quaternion(oR1).coeffs(),qa])
vq = np.concatenate([ np.zeros(3),w,qda ])
aq = np.concatenate([ oR1.T@o_a,w,qdda ])
tauq = np.concatenate([ np.zeros(6),taua ])
#a0 = np.zeros(rm.nv)


total_torques = pin.rnea(rm,rd,q,vq,aq)-tauq


### Check that the computation does not depend on linear position and linear velocity
q[:3] = np.random.rand(3)
vq[:3] = np.random.rand(3)
aq[:3] += np.cross(vq[:3],w)

check = pin.rnea(rm,rd,q,vq,aq)-tauq
assert(norm(check-total_torques)<1e-6)

### Check that the only R dependancy is gravity
q[3:7] = [0,0,0,1]
q[:3] = 0
aq[:3] -= np.cross(vq[:3],w)
vq[:3] = 0

rm.gravity = pin.Motion.Zero()
M = pin.crba(rm,rd,q)
check = pin.rnea(rm,rd,q,vq,aq) + M@np.concatenate([oR1.T@rm.gravity.linear,np.zeros(rm.nv-3)])-tauq


