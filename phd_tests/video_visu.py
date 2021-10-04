import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin
import time
import sys
from os.path import join
from pinocchio.visualize import GepettoVisualizer

def GepettoViewerServer(windowName="python-pinocchio", sceneName="world", loadModel=False):
    """
    Init gepetto-viewer by loading the gui and creating a window.
    """
    import gepetto.corbaserver
    try:
        viewer = gepetto.corbaserver.Client()
        gui = viewer.gui

        # Create window
        window_l = gui.getWindowList()
        if windowName not in window_l:
            gui.windowID = gui.createWindow(windowName)
        else:
            gui.windowID = gui.getWindowID(windowName)

        # Create scene if needed
        scene_l = gui.getSceneList()
        if sceneName not in scene_l:
            gui.createScene(sceneName)
            gui.addSceneToWindow(sceneName, gui.windowID)

        gui.sceneName = sceneName

        return gui

    except Exception:
        print("Error while starting the viewer client. ")
        print("Check whether gepetto-gui is properly started")


if __name__=='__main__':

    alias = sys.argv[1]
    df_kf = pd.read_csv(f'/home/cdebeunne/wolf/objectslam/demos/results_{alias}/result.csv', header=0)
    df_map = pd.read_csv(f'/home/cdebeunne/wolf/objectslam/demos/results_{alias}/init.csv')

    # Scale
    scale = 1.00

    gui = GepettoViewerServer(windowName="tuto")
    gui.deleteNode('world', True)
    gui = GepettoViewerServer(windowName="tuto")

    max_lmk = 0

    for index, row in df_kf.iterrows():
        gui.addSphere("world/frame"+str(index), .01, [1, 0, 0, 1])
        gui.applyConfiguration("world/frame"+str(index), [row['px']*scale, row['py']*scale, row['pz']*scale, row['qx'], row['qy'], row['qz'], row['qw']])
        wRc = R.from_quat([df_kf['qx'][0], df_kf['qy'][0], df_kf['qz'][0], df_kf['qw'][0]]).as_matrix()
        wtc = np.array([row['px'], row['py'], row['pz']])
        wMc = np.identity(4)
        wMc[0:3,0:3] = wRc
        wMc[0:3,3] = wtc
        wMc = pin.SE3(wMc)

        gui.refresh()

        ts = row['t']
        idx_min_ts = (df_map['t'] - ts).abs().idxmin()
        ts = df_map['t'][idx_min_ts]
        kf_map = df_map.loc[df_map['t'] == ts]
        counter = 0
        for index, row in kf_map.iterrows():
            gui.addBox('world/box'+str(counter), 0.05,0.05,0.07,  [.5, .5, 1, 1])
            #gui.addBox('world/box'+str(counter), .3,1.0,0.07,  [.5, .5, 1, 1])
            gui.applyConfiguration("world/box"+str(counter), [row['px'], row['py'], row['pz'], row['qx'], row['qy'], row['qz'], row['qw']])
            gui.refresh()
            counter +=1
        
        if (counter > max_lmk):
            max_lmk = counter
        else:
            for i in range(counter, max_lmk+1):
                gui.addBox('world/box'+str(counter), 0.05,0.05,0.07,  [.5, .5, 1, 1])
                #gui.addBox('world/box'+str(counter), .3,1.0,0.07,  [.5, .5, 1, 1])
                gui.applyConfiguration("world/box"+str(counter), [0, 0, 2.0, row['qx'], row['qy'], row['qz'], row['qw']])
                gui.refresh()

        time.sleep(0.08)

    df_gt = pd.read_pickle(f'groundtruth_{alias}.pkl')
	# rescale gt timestamp
    df_gt['cmTimestamp'] = df_gt['cmTimestamp']-df_gt['cmTimestamp'][0]
    # Loading calibration data for the camera
    calibration = np.load('calibration_cam.npz')
    cmMc = pin.SE3(calibration['cm_M_c'])

    # Calibration with the first frame
    calibration_wolf = np.load('calibration_wolf.npz')
    mMw = pin.SE3(calibration_wolf['mMw'])
    for index, row in df_kf.iterrows():
        # Select the mocap frame
        ts_kf = row['t']
        idxCm = df_gt['cmTimestamp'].sub(float(ts_kf)).abs().idxmin()
        mMcm = pin.SE3(df_gt['mTcm'][idxCm])
        wMc = mMw.inverse() * mMcm * cmMc
        r = R.from_matrix(wMc.rotation)
        q_mocap = r.as_quat()
        t_mocap = wMc.translation
        
        gui.addSphere("world/mocap"+str(index),0.01, [0, 1, 0, 1])
        gui.applyConfiguration("world/mocap"+str(index), [t_mocap[0], t_mocap[1], t_mocap[2], q_mocap[0], q_mocap[1], q_mocap[2], q_mocap[3]])
        gui.refresh()


    while True:
        time.sleep(10)
        gui.refresh()

