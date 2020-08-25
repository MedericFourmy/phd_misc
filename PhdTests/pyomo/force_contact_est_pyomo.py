#!/usr/bin/env python
# coding: utf-8

import sys
import time
import numpy as np
import pinocchio as pin
from example_robot_data import loadANYmal
import curves
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Binary, Constraint, Objective, minimize, log, SolverFactory




examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/multicontact-api-master-examples/examples/ANYMAL/'
file_name = 'anymal_walk_WB.cs'

cs = ContactSequence()
cs.loadFromBinary(examples_dir + file_name)


q_traj   = cs.concatenateQtrajectories()
dq_traj  = cs.concatenateDQtrajectories()
ddq_traj = cs.concatenateDDQtrajectories()
contact_frames = cs.getAllEffectorsInContact()
f_traj_lst = [cs.concatenateContactForceTrajectories(l) for l in contact_frames]
tau_traj = cs.concatenateTauTrajectories()
print(contact_frames)

min_ts = q_traj.min()
max_ts = q_traj.max()
print('traj dur (s): ', max_ts - min_ts)


dt = 1e-3  # discretization timespan
t_arr   = np.arange(min_ts, 0.1, dt)
q_arr   = np.array([q_traj(t) for t in t_arr])
dq_arr  = np.array([dq_traj(t) for t in t_arr])
ddq_arr = np.array([ddq_traj(t) for t in t_arr])
tau_arr = np.array([tau_traj(t) for t in t_arr])
f_arr_lst = [np.array([f_traj_lst[i](t) for t in t_arr]) for i in range(4)]


N = t_arr.shape[0]  # nb of discretized timestamps


robot = loadANYmal()
rmodel = robot.model
rdata = robot.data
contact_frame_ids = [rmodel.getFrameId(l) for l in contact_frames]

def compute_joint_jac(robot, q, cf_ids, world_frame=True):
    Jlinvel = np.zeros((12, robot.nv))
    for i, frame_id in enumerate(cf_ids):
        if world_frame:
            oTl = robot.framePlacement(q, frame_id, update_kinematics=False)
            Jlinvel[3*i:3*(i+1),:] = oTl.rotation @ robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in world coord
        else: 
            Jlinvel[3*i:3*(i+1),:] = robot.computeFrameJacobian(q, frame_id)[:3,:]  # jac in local coord
    return Jlinvel


########################
# PYOMO model of 
########################

from pyomo.environ import AbstractModel, Var, Param, Reals, NonNegativeIntegers, Binary, \
                          Constraint, Objective, minimize, SolverFactory, \
                          RangeSet, summation

##########################
# Create an abstract model
##########################
model = AbstractModel()

# size of forces vector variable components
model.nf = Param(within=NonNegativeIntegers, initialize=12)
# size of binary contact variable vector
model.nc = Param(within=NonNegativeIntegers, initialize=4)
# size of external general torque vector parameter components
model.ntau = Param(within=NonNegativeIntegers, initialize=12)
# size of square jacobians
model.nJ = Param(within=NonNegativeIntegers, initialize=3)

# index set of force vector (f1x, f1y, f1z, f2x, ..., f3z)
model.fI = RangeSet(0, model.nf-1)
# index set of contacts
model.cI = RangeSet(0, model.nc-1)
# index set of contact force torques
model.taucfI = RangeSet(0, model.ntau-1)
# index of sub jacobians
model.lI = RangeSet(0, model.nJ-1)

# VARIABLES
# create 12D force vector real variable
model.f = Var(model.fI, domain=Reals, initialize=0.0)
# create 4D contact vector binary variable
# model.c = Var(model.cI, domain=Reals, initialize=0, bounds=(0,1))
model.c = Var(model.cI, domain=Binary, initialize=1)

# PARAMETERS
model.taucf = Param(model.taucfI, domain=Reals, mutable=True)


model.Jl0T = Param(model.lI, model.lI, domain=Reals, mutable=True)
model.Jl1T = Param(model.lI, model.lI, domain=Reals, mutable=True)
model.Jl2T = Param(model.lI, model.lI, domain=Reals, mutable=True)
model.Jl3T = Param(model.lI, model.lI, domain=Reals, mutable=True)





# weird way to exppress matrix multiplication...
# def obj_expression(model):
#     return 0

def obj_expression(model):
    return (sum((model.Jl0T[0,j] * model.c[0]*model.f[0+j] - model.taucf[0] )**2 for j in range(3))  
          + sum((model.Jl0T[1,j] * model.c[0]*model.f[0+j] - model.taucf[1] )**2 for j in range(3))
          + sum((model.Jl0T[2,j] * model.c[0]*model.f[0+j] - model.taucf[2] )**2 for j in range(3))
          + sum((model.Jl1T[0,j] * model.c[1]*model.f[3+j] - model.taucf[3] )**2 for j in range(3))
          + sum((model.Jl1T[1,j] * model.c[1]*model.f[3+j] - model.taucf[4] )**2 for j in range(3))
          + sum((model.Jl1T[2,j] * model.c[1]*model.f[3+j] - model.taucf[5] )**2 for j in range(3))
          + sum((model.Jl1T[0,j] * model.c[2]*model.f[6+j] - model.taucf[6] )**2 for j in range(3))
          + sum((model.Jl1T[1,j] * model.c[2]*model.f[6+j] - model.taucf[7] )**2 for j in range(3))
          + sum((model.Jl1T[2,j] * model.c[2]*model.f[6+j] - model.taucf[8] )**2 for j in range(3))
          + sum((model.Jl1T[0,j] * model.c[3]*model.f[9+j] - model.taucf[9] )**2 for j in range(3))
          + sum((model.Jl1T[1,j] * model.c[3]*model.f[9+j] - model.taucf[10])**2 for j in range(3))
          + sum((model.Jl1T[2,j] * model.c[3]*model.f[9+j] - model.taucf[11])**2 for j in range(3)))

model.OBJ = Objective(rule=obj_expression)

# constraints
# positive forces along z
# 
model.c0 = Constraint(rule=lambda minst: minst.f[2] >= 0)  
model.c1 = Constraint(rule=lambda minst: minst.f[5] >= 0)  
model.c2 = Constraint(rule=lambda minst: minst.f[8] >= 0)  
model.c3 = Constraint(rule=lambda minst: minst.f[11] >= 0)  

model.c4 = Constraint(rule=lambda minst: (1 - minst.c[0]) * minst.f[2]  == 0)
model.c5 = Constraint(rule=lambda minst: (1 - minst.c[1]) * minst.f[5]  == 0)
model.c6 = Constraint(rule=lambda minst: (1 - minst.c[2]) * minst.f[8]  == 0)
model.c7 = Constraint(rule=lambda minst: (1 - minst.c[3]) * minst.f[11] == 0)


#Solve the model using MindtPy
solver=SolverFactory('mindtpy')

#################################
# instanciate and fill the values
#################################
instance = model.create_instance()


# retrieve data to setup the problem
q = q_arr[0,:]
dq = dq_arr[0,:]
ddq = ddq_arr[0,:]
tau_j = tau_arr[0,:]
Jlinvel = compute_joint_jac(robot, q, contact_frame_ids)

tau_cf = pin.rnea(rmodel, rdata, q, dq, ddq)[6:]
tau_cf -= tau_j

Jlinvel = compute_joint_jac(robot, q_arr[0,:], contact_frame_ids)
Jlinvel_without_freflyer = Jlinvel[:,6:]


# set the instance data
# not possible to set vector wise
for i in range(12):
    instance.taucf[i] = tau_cf[i]


J0 = Jlinvel_without_freflyer[0:3,0:3].T
J1 = Jlinvel_without_freflyer[3:6,3:6].T
J2 = Jlinvel_without_freflyer[6:9,6:9].T
J3 = Jlinvel_without_freflyer[9:12,9:12].T

for i in range(3):
    for j in range(3):
        instance.Jl0T[i,j] = J0[i,j]
        instance.Jl1T[i,j] = J1[i,j]
        instance.Jl2T[i,j] = J2[i,j]
        instance.Jl3T[i,j] = J3[i,j]




results = solver.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)


# instance.objective.display()
instance.display()
instance.pprint()


