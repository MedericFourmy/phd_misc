import os
import numpy as np
import pandas as pd
import pinocchio as pin
try:
    mcapi_import = True
    from multicontact_api import ContactSequence, ContactPhase, ContactPatch
except ImportError:
    mcapi_import = False
    print('Cannot import multicontact_api, TrajLogger.store_mcapi_traj  will not work!!')
from curves import piecewise

class TrajLogger:

    def __init__(self, contact_names, directory=''):
        self.data_log = {
            't': [],  # time trajectory
            'q': [],  # configuration position trajectory
            'v': [],   # configuration velocity trajectory
            }
        self.contact_names = contact_names
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
    
    def append_data_from_sol(self, t, q, v, dv, tsidw, sol):
        """
        tsidw: TsidQuadruped or TsidBiped(not tested)
        """
        data = {
            't': t,
            'q': q.copy(),
            'v': v.copy(),
            'dv': dv.copy()
        }
        
        for i_foot in range(tsidw.nc):
            if tsidw.invdyn.checkContact(tsidw.contacts[i_foot].name, sol):
                f = tsidw.invdyn.getContactForce(tsidw.contacts[i_foot].name, sol) 
                if tsidw.conf.contact6d:
                    f = tsidw.contacts[i_foot].getForceGeneratorMatrix @ f
            else:
                if tsidw.conf.contact6d:
                    f = np.zeros(6)
                else:
                    f = np.zeros(3)

            data['f{}'.format(i_foot)] = f        
        data['tau'] = tsidw.invdyn.getActuatorForces(sol)

        # in advance of 1 dt wrt q and v
        data['c'] = tsidw.robot.com(tsidw.invdyn.data())
        data['dc'] = tsidw.robot.com_vel(tsidw.invdyn.data())
        data['Lc'] = pin.computeCentroidalMomentum(tsidw.robot.model(), tsidw.robot.data(), q, v).angular
        data['contacts'] = np.array([tsidw.invdyn.checkContact(contact.name, sol) for contact in tsidw.contacts])

        self.append_data(data)

    def set_data_lst_as_arrays(self):
        for key in self.data_log:
            self.data_log[key] = np.array(self.data_log[key])
    
    def store_csv_trajs(self, traj_name, sep, skip_free_flyer=True, time_sec=False):
        self.set_data_lst_as_arrays()
        q_traj = self.data_log['q'].copy()
        v_traj = self.data_log['v'].copy()
        tau_traj = self.data_log['tau'].copy()
        contacts_traj = self.data_log['contacts'].copy() * 1  # * 1 to store 1 and 0s instead of False True
        assert(q_traj.shape[0] == v_traj.shape[0] == tau_traj.shape[0] == contacts_traj.shape[0])
        N = q_traj.shape[0]
        print('traj size: ', N)

        df_q = pd.DataFrame()
        df_v = pd.DataFrame()
        df_tau = pd.DataFrame()
        df_contacts = pd.DataFrame()

        if skip_free_flyer:
            q_traj = q_traj[:,7:]
            v_traj = v_traj[:,6:]

        if time_sec:
            df_q['t'] = self.data_log['t']
            df_v['t'] = self.data_log['t']
            df_tau['t'] = self.data_log['t']
            df_contacts['t'] = self.data_log['t']
        else:
            df_q['t'] = np.arange(N)
            df_v['t'] = np.arange(N)
            df_tau['t'] = np.arange(N)
            df_contacts['t'] = np.arange(N)
            
        for col in range(q_traj.shape[1]):
            colname = 'q{}'.format(col)
            df_q[colname] = q_traj[:,col]
        for col in range(v_traj.shape[1]):
            colname = 'v{}'.format(col)
            df_v[colname] = v_traj[:,col]
        for col in range(tau_traj.shape[1]):
            colname = 'tau{}'.format(col)
            df_tau[colname] = tau_traj[:,col]
        for col, cname in enumerate(self.contact_names):
            df_contacts[cname] = contacts_traj[:,col]
        
        df_q.to_csv(os.path.join(self.directory, '{}_q.dat'.format(traj_name)), sep=sep, index=False, header=False)
        df_v.to_csv(os.path.join(self.directory, '{}_v.dat'.format(traj_name)), sep=sep, index=False, header=False)
        df_tau.to_csv(os.path.join(self.directory, '{}_tau.dat'.format(traj_name)), sep=sep, index=False, header=False)
        df_contacts.to_csv(os.path.join(self.directory, '{}_contacts.dat'.format(traj_name)), sep=sep, index=False, header=False)
        df_contact_names = pd.DataFrame(columns=self.contact_names)
        df_contact_names.to_csv(os.path.join(self.directory, '{}_contact_names.dat'.format(traj_name)), sep=sep, index=False, header=True)

        print('q, v, tau and contacts traj .dat files saved in: ', self.directory)

    def store_mcapi_traj(self, traj_name):
        if not mcapi_import:
            print('multicontact_api package import has failed, check your install')
            return
            
        self.set_data_lst_as_arrays()

        # trajectory with only one ContactPhase (simpler to read/write)
        # when feet not in contact, the force is exactly zero, that's the only diff
        cs = ContactSequence()
        cp = ContactPhase()

        # assign trajectories :
        t_arr = self.data_log['t']

        cp.timeInitial = t_arr[0]
        cp.timeFinal = t_arr[-1]
        cp.duration = t_arr[-1] - t_arr[0] 

        # col number of trajectories should be the time traj size hence the transpose
        cp.q_t = piecewise.FromPointsList(self.data_log['q'].T, t_arr)
        cp.dq_t = piecewise.FromPointsList(self.data_log['v'].T, t_arr)
        cp.ddq_t = piecewise.FromPointsList(self.data_log['dv'].T, t_arr)
        cp.tau_t = piecewise.FromPointsList(self.data_log['tau'].T, t_arr)
        cp.c_t = piecewise.FromPointsList(self.data_log['c'].T, t_arr)
        cp.dc_t = piecewise.FromPointsList(self.data_log['dc'].T, t_arr)
        cp.L_t = piecewise.FromPointsList(self.data_log['Lc'].T, t_arr)

        # contact force trajectories
        for i_foot, frame_name in enumerate(self.contact_names):
            cp.addContact(frame_name, ContactPatch(pin.SE3(),0.5))  # dummy placement and friction coeff
            cp.addContactForceTrajectory(frame_name, piecewise.FromPointsList(self.data_log['f{}'.format(i_foot)].T, t_arr))

        cs.append(cp)  # only one contact phase

        savepath = os.path.join(self.directory, traj_name+'.cs')
        cs.saveAsBinary(savepath)
        print('Saved ' + savepath)

