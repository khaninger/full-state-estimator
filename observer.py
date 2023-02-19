import casadi as ca
import numpy as np
from robot import robot

class ekf():
    """ This defines an EKF observer """
    def __init__(self, par, q0, est_geom = False, est_stiff = False):
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state
        self.est_geom = est_geom
        self.est_stiff = est_stiff
        est_par = {}

        if est_geom:
            est_par = {'pos':ca.SX.sym('contact_1_pos',3)}
            self.x['p'] = 0.1*ca.DM.ones(3)
        if est_stiff:
            est_par = {'stiff':ca.SX.sym('contact_1_stiff',3)}
            self.x['stiff'] = 1000*ca.DM.ones(3)
        
        self.dyn_sys = robot(par, est_par = est_par)
        self.proc_noise = np.diag(np.concatenate((par['proc_noise']['pos'],
                                                  par['proc_noise']['vel'],
                                                  par['proc_noise'].get('geom', []),
                                                  par['proc_noise'].get('stiff', []))))

        self.cov = np.diag(par['cov_init'])  # initial covariance
        self.meas_noise = np.diag(par['meas_noise']['pos'])

    def step(self, q, tau, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau':tau,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'],
                                     self.x.get('p', []),
                                     self.x.get('stiff', []))}
        x_next = self.dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        #print(f"F_i = {self.dyn_sys.jacpinv(self.x['q']).T@step_args['tau']}")
        #print(f"tau_err = {x_next['tau_err']}")
        #print(f"tau     = {tau}")
        #print(step_args['tau_err'])
        #print(q-x_next['xi_next'][:6])
        
        cov_next = A@self.cov@(A.T) + self.proc_noise
        L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(L)):
           raise ValueError("Nans in the L matrix")
        xi_corr = x_next['xi_next'] + L@(q - x_next['xi_next'][:self.dyn_sys.nq])
        self.x['q'] = xi_corr[:self.dyn_sys.nq].full()
        self.x['dq'] = xi_corr[self.dyn_sys.nq:2*self.dyn_sys.nq].full()
        if self.est_geom: self.x['p'] = xi_corr[2*self.dyn_sys.nq:].full()
        if self.est_stiff: self.x['stiff'] = xi_corr[2*self.dyn_sys.nq:].full().flatten()
        self.x['cont_pt'] = x_next['cont_pt'].full().flatten()
        x_ee = self.dyn_sys.fwd_kin(self.x['q'])
        self.x['x_ee'] = (x_ee[0].full(),
                          x_ee[1].full())
        self.x['xi'] = xi_corr.full()
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-L@C)@cov_next
        return self.x

    def likelihood(self, obs):
        return NotImplemented

class momentum_obs():
    ''' Mostly following https://elib.dlr.de/129060/1/root.pdf '''
    
    def __init__(self, par, q0):
        self.par = par
        self.K = 20*np.eye(q0.size) # gain of momentum obs, dim N
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state
        self.p_last = np.zeros(q0.size)
        self.r = np.zeros(q0.size)
        
    def step(self, q, tau, F = None):
        step_args = {'tau':tau,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'])}
        x_next = self.dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step        
        
        r += self.par['K']*(x_next['mom']-self.p_last-\
                            self.par['h']*(x_next['tau_err']))
        self.dyn_sys.jacpinv(self.x['q']).T@step_args['tau']
