import os
import numpy as np
import pandas as pd
import pinocchio as pin
from multicontact_api import ContactSequence, ContactPhase, ContactPatch
from curves import piecewise

class TrajLogger:

    def __init__(self, directory=''):
        self.data_log = {
            't': [],  # time trajectory
            'q': [],  # configuration position trajectory
            'v': []   # configuration velocity trajectory
            }
        self.directory = directory

    def append_data(self, data_dic):
        """
        data_dic: new data to append to the log

        Only mandatory fields are those described in the constructor.
        Datas must be stored as numpy arrays.
        """
        assert all (k in data_dic for k in 'tqv')
        for key, val in data_dic.items():
            if key not in self.data_log:
                self.data_log[key] = [val]
            else:
                self.data_log[key].append(val)
    
    def append_data_from_sol(self, t, q, v, dv, tsid_wrapper, sol):
        """
        tsid_wrapper: TsidQuadruped or TsidBiped(not tested)
        """
        data = {
            't': t,
            'q': q,
            'v': v,
            'dv': dv,
        }
        
        for i_foot in range(4):
            if tsid_wrapper.invdyn.checkContact(tsid_wrapper.contacts[i_foot].name, sol):
                f = tsid_wrapper.invdyn.getContactForce(tsid_wrapper.contacts[i_foot].name, sol) 
            else:
                f = np.zeros(3)
            data['f{}'.format(i_foot)] = f        
        data['tau'] = tsid_wrapper.invdyn.getActuatorForces(sol)

        data['c'] = tsid_wrapper.robot.com(tsid_wrapper.invdyn.data())
        data['dc'] = tsid_wrapper.robot.com_vel(tsid_wrapper.invdyn.data())
        data['ddc'] = tsid_wrapper.comTask.getAcceleration(dv)
        data['Lc'] = pin.computeCentroidalMomentum(tsid_wrapper.robot.model(), tsid_wrapper.robot.data(), q, v).angular

        self.append_data(data)

    def set_data_lst_as_arrays(self):
        for key in self.data_log:
            self.data_log[key] = np.array(self.data_log[key])
    
    def store_qv_trajs(self, traj_name, sep, skip_free_flyer=True, time_sec=False):
        if isinstance(self.data_log['q'], list): self.data_log['q'] = np.array(self.data_log['q']) 
        if isinstance(self.data_log['v'], list): self.data_log['v'] = np.array(self.data_log['v']) 
        q_traj = self.data_log['q'].copy()
        v_traj = self.data_log['v'].copy()
        df_q = pd.DataFrame()
        df_v = pd.DataFrame()
        if time_sec:
            df_q['t'] = np.array(self.data_log['t'])
            df_v['t'] = np.array(self.data_log['t'])
        else:
            df_q['t'] = np.arange(q_traj.shape[0])
            df_v['t'] = np.arange(v_traj.shape[0])
        if skip_free_flyer:
            q_traj = q_traj[:,7:]
            v_traj = v_traj[:,6:]
        
        for col in range(q_traj.shape[1]):
            colname = 'q{}'.format(col)
            df_q[colname] = q_traj[:,col]
        for col in range(v_traj.shape[1]):
            colname = 'v{}'.format(col)
            df_v[colname] = v_traj[:,col]
        
        df_q.to_csv (os.path.join(self.directory, '{}_q.csv'.format(traj_name)), sep=sep, index=False, header=False)
        df_v.to_csv (os.path.join(self.directory, '{}_v.csv'.format(traj_name)), sep=sep, index=False, header=False)
        print('q and v traj saved: ')

    def store_mcapi_traj(self, tsid_wrapper, traj_name):
        # trajectory with only one ContactPhase (simpler to read/write)
        # when feet not in contact, the force is exactly zero, that's the only diff

        cs = ContactSequence()
        cp = ContactPhase()

        # assign trajectories :
        t_arr = self.data_log['t']

        cp.timeInitial = t_arr[0]
        cp.timeFinal = t_arr[-1]
        cp.duration = t_arr[-1] - t_arr[0] 

        print(t_arr.shape)
        print(self.data_log['q'].shape)
        print(self.data_log['v'].shape)
        print(self.data_log['dv'].shape)
        print(self.data_log['tau'].shape)
        
        # col number of trajectories should be the time traj size hence the transpose
        cp.q_t = piecewise.FromPointsList(self.data_log['q'].T, t_arr)
        cp.dq_t = piecewise.FromPointsList(self.data_log['v'].T, t_arr)
        cp.ddq_t = piecewise.FromPointsList(self.data_log['dv'].T, t_arr)
        cp.tau_t = piecewise.FromPointsList(self.data_log['tau'].T, t_arr)
        cp.c_t = piecewise.FromPointsList(self.data_log['c'].T, t_arr)
        cp.dc_t = piecewise.FromPointsList(self.data_log['dc'].T, t_arr)
        cp.ddc_t = piecewise.FromPointsList(self.data_log['ddc'].T, t_arr)  # not needed
        cp.L_t = piecewise.FromPointsList(self.data_log['Lc'].T, t_arr)
        # cp.wrench_t = wrench
        # cp.zmp_t = zmp
        # cp.root_t = root

        # contact force trajectories
        for i_foot, frame_name in enumerate(tsid_wrapper.foot_frame_names):
            cp.addContact(frame_name, ContactPatch(pin.SE3(),0.5))  # dummy placement and friction coeff
            cp.addContactForceTrajectory(frame_name, piecewise.FromPointsList(self.data_log['f{}'.format(i_foot)].T, t_arr))

        cs.append(cp)  # only one contact phase

        savepath = os.path.join(self.directory, traj_name+'.cs')
        cs.saveAsBinary(savepath)
        print('Saved ' + savepath)

