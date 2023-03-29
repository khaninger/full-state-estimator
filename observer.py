import casadi as ca
import numpy as np
from robot import robot
from scipy.stats import multivariate_normal
import copy

def static_ekf_update(cov, x, q, tau, dyn_sys, proc_noise, meas_noise, est_geom, est_stiff, F=None):
    step_args = {'tau':tau,
                 'xi':ca.vertcat(x['q'], x['dq'],
                                     x.get('p', []),
                                     x.get('stiff', []))}
    x_next = dyn_sys.disc_dyn.call(step_args)
    #print(x_next)
    x_next = x_next["xi_next"]
    #print(x_next)
    A, C = dyn_sys.get_linearized(step_args)  # get the linearized dynamics and observation matrices
    cov_next = A @ cov @ (A.T) + proc_noise
    S = C @ cov_next @ (C.T) + meas_noise
    y_hat = C @ x_next
    L = cov_next @ C.T @ ca.inv(C @ cov_next @ (C.T) + meas_noise)  # calculate Kalman gain
    if np.any(np.isnan(L)): raise ValueError("Nans in the L matrix")
    xi_corr = x_next['xi_next'] + L @ (q - x_next['xi_next'][dyn_sys.nq])
    x['q'] = xi_corr[:dyn_sys.nq].full()
    x['dq'] = xi_corr[dyn_sys.nq:2 * dyn_sys.nq].full()
    if est_geom: x['p'] = xi_corr[2 * dyn_sys.nq:].full()
    if est_stiff: x['stiff'] = xi_corr[2 * dyn_sys.nq:].full().flatten()
    x['cont_pt'] = x_next['cont_pt'].full().flatten()
    x_ee = dyn_sys.fwd_kin(x['q'])
    x['x_ee'] = (x_ee[0].full(),
                      x_ee[1].full())
    x['xi'] = xi_corr.full()
    mom_obs.step(x_next['mom'], x_next['tau_err'])
    x['f_ee_mo'] = (dyn_sys.jacpinv(x['q']) @ mom_obs.r).full()
    x['f_ee_obs'] = -(dyn_sys.jacpinv(x['q']) @ x_next['tau_i']).full()

    cov = (ca.DM.eye(dyn_sys.nx) - L @ C) @ cov_next  # corrected covariance

    return x, cov, S, y_hat

class ekf():
    """ This defines an EKF observer """
    def __init__(self, par, q0, est_geom = False, est_stiff = False):
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state
        self.est_geom = est_geom
        self.est_stiff = est_stiff
        
        self.est_par = {}
        if est_geom:
            self.est_par = {'pos':ca.SX.sym('contact_1_pos',3)}
            self.x['p'] = 0.15*ca.DM.ones(3)
        if est_stiff:
            self.est_par = {'stiff':ca.SX.sym('contact_1_stiff',3)}
            self.x['stiff'] = 1*ca.DM.ones(3)

        #self.dyn_sys = robot(par, est_par = est_par)
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


        
    def step(self, q, tau, dyn_sys, F = None):
        """ Steps the observer based on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau':tau,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'],
                                     self.x.get('p', []),
                                     self.x.get('stiff', []))}
        #print(self.cov)
        self.x_next = dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step

        x_next = self.x_next["xi_next"]
        #print(x_next.shape)
        A, C = dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        #print(C.shape)
        #print(f"F_i = {self.dyn_sys.jacpinv(self.x['q']).T@x_next['tau_err']}")
        #print(f"tau_err = {x_next['tau_err']}")
        #print(f"tau     = {tau}")
        #print(step_args['tau_err'])
        #print(q-x_next['xi_next'][:6])
        #print(A)
        #print(np.argwhere(np.isnan(self.cov)))
        self.cov_next = A@self.cov@(A.T) + self.proc_noise
        #print(C)

        #print(self.cov)
        self.S = C@self.cov_next@(C.T) + self.meas_noise
        self.y_hat = C@x_next
        #print(self.y_hat.shape)
        #print(self.cov_next)
        self.L = self.cov_next@C.T@ca.inv(C@self.cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        #print(self.L.shape)
        if np.any(np.isnan(self.L)): raise ValueError("Nans in the L matrix")
    
        xi_corr = self.x_next['xi_next'] + self.L@(q - self.x_next['xi_next'][dyn_sys.nq])
        #print((L@(q - x_next['xi_next'][:self.dyn_sys.nq]))[-3:])
        #print(xi_corr)
        self.x['q'] = xi_corr[:dyn_sys.nq].full()
        self.x['dq'] = xi_corr[dyn_sys.nq:2*dyn_sys.nq].full()
        if self.est_geom: self.x['p'] = xi_corr[2*dyn_sys.nq:].full()
        if self.est_stiff: self.x['stiff'] = xi_corr[2*dyn_sys.nq:].full().flatten()
        self.x['cont_pt'] = self.x_next['cont_pt'].full().flatten()
        x_ee = dyn_sys.fwd_kin(self.x['q'])
        self.x['x_ee'] = (x_ee[0].full(),
                          x_ee[1].full())
        self.x['xi'] = xi_corr.full()
        self.mom_obs.step(self.x_next['mom'], self.x_next['tau_err'])
        self.x['f_ee_mo'] = (dyn_sys.jacpinv(self.x['q'])@self.mom_obs.r).full()
        self.x['f_ee_obs'] = -(dyn_sys.jacpinv(self.x['q'])@self.x_next['tau_i']).full()
        
        self.cov = (ca.DM.eye(dyn_sys.nx)-self.L@C)@self.cov_next  # corrected covariance
        #print("debug2")
        #print(self.cov)
        #print(self.cov[-3:,-3:])
        #x_est = copy.deepcopy(self.x)
        #cov_est = copy.deepcopy(self.cov)
        #S = copy.deepcopy(self.S)
        #y_hat = copy.deepcopy(self.y_hat)
        return self.x, self.cov, self.S, self.y_hat

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
