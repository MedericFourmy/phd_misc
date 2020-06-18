import numpy as np
import pandas as pd
import pinocchio as pin

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
        
        df_q.to_csv ('{}_q.csv'.format(traj_name), sep=sep, index=False, header=False)
        df_v.to_csv ('{}_v.csv'.format(traj_name), sep=sep, index=False, header=False)

    def store_mcapi_traj(self):
        pass