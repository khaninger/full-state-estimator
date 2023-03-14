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
            self.x['p'] = 0.15*ca.DM.ones(3)
        if est_stiff:
            est_par = {'stiff':ca.SX.sym('contact_1_stiff',3)}
            self.x['stiff'] = 1*ca.DM.ones(3)

        self.dyn_sys = robot(par, est_par = est_par)
        self.proc_noise = np.diag(np.concatenate((par['proc_noise']['pos'],
                                                  par['proc_noise']['vel'],
                                                  par['proc_noise']['geom'] if est_geom else [],
                                                  par['proc_noise']['stiff'] if est_stiff else [])))

        
        self.cov = np.diag(np.concatenate((par['cov_init']['pos'],
                                           par['cov_init']['vel'],
                                           par['cov_init']['geom'] if est_geom else [],
                                           par['cov_init']['stiff'] if est_stiff else [])))
        
        self.meas_noise = np.diag(par['meas_noise']['pos'])
        self.mom_obs = MomentumObserver(par, q0)
        
    def step(self, q, tau, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau':tau,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'],
                                     self.x.get('p', []),
                                     self.x.get('stiff', []))}
        #print(self.cov)
        x_next = self.dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        #print(f"F_i = {self.dyn_sys.jacpinv(self.x['q']).T@x_next['tau_err']}")
        #print(f"tau_err = {x_next['tau_err']}")
        #print(f"tau     = {tau}")
        #print(step_args['tau_err'])
        #print(q-x_next['xi_next'][:6])
        #print(A)
        cov_next = A@self.cov@(A.T) + self.proc_noise
        #print(cov_next)
        self.L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(self.L)): raise ValueError("Nans in the L matrix")
    
        xi_corr = x_next['xi_next'] + self.L@(q - x_next['xi_next'][:self.dyn_sys.nq])
        #print((L@(q - x_next['xi_next'][:self.dyn_sys.nq]))[-3:])
        #print(xi_corr)
        self.x['q'] = xi_corr[:self.dyn_sys.nq].full()
        self.x['dq'] = xi_corr[self.dyn_sys.nq:2*self.dyn_sys.nq].full()
        if self.est_geom: self.x['p'] = xi_corr[2*self.dyn_sys.nq:].full()
        if self.est_stiff: self.x['stiff'] = xi_corr[2*self.dyn_sys.nq:].full().flatten()
        self.x['cont_pt'] = x_next['cont_pt'].full().flatten()
        x_ee = self.dyn_sys.fwd_kin(self.x['q'])
        self.x['x_ee'] = (x_ee[0].full(),
                          x_ee[1].full())
        self.x['xi'] = xi_corr.full()
        self.mom_obs.step(x_next['mom'], x_next['tau_err'])
        self.x['f_ee_mo'] =  (self.dyn_sys.jacpinv(self.x['q'])@self.mom_obs.r).full()
        self.x['f_ee_obs'] = -(self.dyn_sys.jacpinv(self.x['q'])@x_next['tau_i']).full()
        
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-self.L@C)@cov_next # corrected covariance
        #print(self.cov[-3:,-3:])
        return self.x

    def likelihood(self, obs):
        return NotImplemented

class MomentumObserver():
    ''' Mostly following https://elib.dlr.de/129060/1/root.pdf '''
    
    def __init__(self, par, q0):
        self.K = par['mom_obs_K']
        self.h = par['h']
        self.r = np.zeros(q0.size)
        
    def step(self, p, tau_err, F = None):      
        self.r += self.K*(p-self.h*(self.r-tau_err))
