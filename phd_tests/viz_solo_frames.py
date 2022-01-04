import numpy as np
import pinocchio as pin
from example_robot_data import load

robot = load('solo12')
robot.initViewer(loadModel=True)
q0 = robot.model.referenceConfigurations['standing']
# q0[7+3+2] -= 0.5
robot.display(q0)
gv = robot.viewer.gui
window_id = 'python-pinocchio'
print(gv.getCameraTransform(window_id))

cam_pose = [0.9411491751670837, -1.712505578994751, 0.4990329444408417, 0.6610969305038452, 0.14833980798721313, 0.19962812960147858, 0.707880437374115]
# gv.setCameraTransform(window_id, cam_pose)

###### 
# Frames
i_pose_im = [-0.01116, -0.00732, -0.00248, 
            0.00350, 0.00424, 0.0052, 0.9999]  # imu mocap IRI good calib
# IMU MOCAP extr
m_p_mb = np.array([-0.11872364, -0.0027602,  -0.01272801])
m_q_b = np.array([0.00698701, 0.00747303, 0.00597298, 0.99992983])

i_T_m = pin.XYZQUATToSE3(i_pose_im)
m_T_b = pin.XYZQUATToSE3(np.concatenate([m_p_mb, m_q_b]))
o_T_b = pin.XYZQUATToSE3(q0[:7])

o_T_m = o_T_b*m_T_b.inverse()
o_T_i = o_T_b*(i_T_m*m_T_b).inverse()
######

# Scenography
gv.addFloor('world/floor')
gv.setLightingMode('world/floor', 'OFF')
gv.setColor('world/floor', [0.95,0.95,0.95,1])

w = 10  # rectangle half width
gv.addSquareFace('world/faceH', 
                 [-3,  w,  w],
                 [-3, -w,  w],
                 [-3, -w, -w],
                 [-3,  w, -w],
                 [1,1,1,1])
gv.setLightingMode('world/faceH', 'OFF')
gv.addSquareFace('world/faceL', 
                 [ w, 3,  w],
                 [-w, 3,  w],
                 [-w, 3, -w],
                 [ w, 3, -w],
                 [1,1,1,1])
gv.setLightingMode('world/faceH', 'OFF')


LEGS = ['FL', 'FR', 'HL', 'HR']
contacts = [leg+'_ANKLE' for leg in LEGS]
for leg in LEGS:
    gv.setColor(f'world/pinocchio/visuals/base_link_0',[0,0,0,.2])
    gv.setColor(f'world/pinocchio/visuals/{leg}_SHOULDER_0',[0,0,0,.2])
    gv.setColor(f'world/pinocchio/visuals/{leg}_UPPER_LEG_0',[0,0,0,.2])
    gv.setColor(f'world/pinocchio/visuals/{leg}_FOOT_0',[0,0,0,.2])
    gv.setColor(f'world/pinocchio/visuals/{leg}_LOWER_LEG_0',[0,0,0,.2])

robot.forwardKinematics(q0)

main_frame_size = 0.008
main_frame_length = 0.05
second_frame_size = 0.005
second_frame_length = 0.03

gv.addXYZaxis('world/base',[1.,0,0,1], main_frame_size, main_frame_length)
gv.applyConfiguration('world/base', pin.SE3ToXYZQUAT(o_T_b).tolist())

gv.addXYZaxis('world/imu',[1.,0,0,1], main_frame_size, main_frame_length)
gv.applyConfiguration('world/imu', pin.SE3ToXYZQUAT(o_T_i).tolist())

# gv.addXYZaxis('world/mocap',[1.,0,0,1], main_frame_size, main_frame_length)
# gv.applyConfiguration('world/mocap', pin.SE3ToXYZQUAT(o_T_m).tolist())

gv.addXYZaxis('world/world_frame',[1.,0,0,1], main_frame_size, main_frame_length)
gv.applyConfiguration('world/world_frame', [0,-0.3,0.005, 0,0,0,1])

for i in range(4):
    fid = robot.model.getFrameId(contacts[i])
    o_M_l = robot.framePlacement(q0, fid, update_kinematics=True)
    name = f'world/leg{LEGS[i]}'
    gv.addXYZaxis(name,[1.,0,0,1],second_frame_size,second_frame_length)
    gv.applyConfiguration(name, pin.SE3ToXYZQUAT(o_M_l).tolist())

    