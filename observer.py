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
        self.x['xi'] = ca.vertcat(self.x['q'], self.x['dq'],
                                  self.x.get('p', []),
                                  self.x.get('stiff', []))

        self.dyn_sys = robot(par, est_par = est_par)
        self.proc_noise = np.diag(np.concatenate((par['proc_noise']['pos'],
                                                  par['proc_noise']['vel'],
                                                  par['proc_noise']['geom'] if est_geom else [],
                                                  par['proc_noise']['stiff'] if est_stiff else [])))

        
        self.x['cov'] = np.diag(np.concatenate((par['cov_init']['pos'],
                                           par['cov_init']['vel'],
                                           par['cov_init']['geom'] if est_geom else [],
                                           par['cov_init']['stiff'] if est_stiff else [])))
        
        self.meas_noise = np.diag(par['meas_noise']['pos'])

        self.build_step_fn()
        self.mom_obs = MomentumObserver(par, q0)

    def build_step_fn(self):
        # Build a static KF update w arguments of mu, sigma, tau and measured q
        tau = self.dyn_sys.vars['tau']
        mu = self.dyn_sys.vars['xi']
        mu_next = self.dyn_sys.vars['xi_next']
        A, C = self.dyn_sys.get_linearized_opt({'tau':tau, 'xi':mu})
        
        q_meas = ca.SX.sym('q_meas', self.dyn_sys.nq)
        cov = ca.SX.sym('cov', mu.shape[0], mu.shape[0])
        
        cov_next = A@cov@(A.T) + self.proc_noise
        L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain

        mu_next_corr = mu_next + L@(q_meas - mu_next[:self.dyn_sys.nq])
        cov_next_corr = (ca.SX.eye(self.dyn_sys.nx)-L@C)@cov_next # corrected covariance

        fn_dict = {'tau':tau, 'mu':mu, 'cov':cov, 'q_meas':q_meas,
                   'mu_next':mu_next_corr, 'cov_next':cov_next_corr}
        self.step_fn = ca.Function('ekf_step', fn_dict,
                                   ['tau', 'mu', 'cov', 'q_meas'],
                                   ['mu_next', 'cov_next'])

    def step_fast(self, q, tau, F=None):
        step_args = {'tau':tau,
                     'mu':self.x['xi'],
                     'cov':self.x['cov'],
                     'q_meas':q}
        res = self.step_fn.call(step_args)
        self.x['xi'] = res['mu_next'].full()
        self.x['cov'] = res['cov_next']
        return self.x

    
    def step(self, q, tau, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau':tau,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'],
                                     self.x.get('p', []),
                                     self.x.get('stiff', []))}
        
        x_next = self.dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        cov_next = A@self.x['cov']@(A.T) + self.proc_noise
        #print(cov_next)
        self.L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(self.L)): raise ValueError("Nans in the L matrix")
    
        xi_corr = x_next['xi_next'] + self.L@(q - x_next['xi_next'][:self.dyn_sys.nq])
        self.x['xi'] = xi_corr.full()
        self.x['q'] = xi_corr[:self.dyn_sys.nq].full()
        self.x['dq'] = xi_corr[self.dyn_sys.nq:2*self.dyn_sys.nq].full()
        if self.est_geom: self.x['p'] = xi_corr[2*self.dyn_sys.nq:].full()
        if self.est_stiff: self.x['stiff'] = xi_corr[2*self.dyn_sys.nq:].full().flatten()

        
        
        #self.x['cont_pt'] = x_next['cont_pt'].full().flatten()
        #x_ee = self.dyn_sys.fwd_kin(self.x['q'])
        #self.x['x_ee'] = (x_ee[0].full(),
        #                  x_ee[1].full())
        #self.mom_obs.step(x_next['mom'], x_next['tau_err'])
        #self.x['f_ee_mo'] =  (self.dyn_sys.jacpinv(self.x['q'])@self.mom_obs.r).full()
        #self.x['f_ee_obs'] = -(self.dyn_sys.jacpinv(self.x['q'])@x_next['tau_i']).full()
        
        self.x['cov'] = (ca.DM.eye(self.dyn_sys.nx)-self.L@C)@cov_next # corrected covariance
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
