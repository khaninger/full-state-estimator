import casadi as ca
import numpy as np
from robot import Robot
from scipy.stats import multivariate_normal
import copy

def build_step_fn(robot):
    # Build a static KF update w arguments of mu, sigma, tau and measured q
    proc_noise = robot.proc_noise
    meas_noise = robot.meas_noise

    tau = robot.vars['tau']
    mu = robot.vars['xi']
    mu_next = robot.vars['xi_next']
    A, C = robot.get_linearized({'tau':tau, 'xi':mu})

    q_meas = ca.SX.sym('q_meas', robot.nq)
    cov = ca.SX.sym('cov', mu.shape[0], mu.shape[0])
    
    cov_next = A@cov@(A.T) + proc_noise
    L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + meas_noise) # calculate Kalman gain

    mu_next_corr = mu_next + L@(q_meas - mu_next[:robot.nq])
    cov_next_corr = (ca.SX.eye(robot.nx)-L@C)@cov_next # corrected covariance

    fn_dict = {'tau':tau, 'mu':mu, 'cov':cov, 'q_meas':q_meas,
               'mu_next':mu_next_corr, 'cov_next':cov_next_corr}
    
    step_fn = ca.Function('ekf_step', fn_dict,
                          ['tau', 'mu', 'cov', 'q_meas'], # inputs to casadi function
                          ['mu_next', 'cov_next'])        # outputs of casadi function
    return step_fn

class ekf():
    """ This defines an EKF observer """
    def __init__(self, robot):
        self.x = {'mu':robot.xi_init, 'cov':robot.cov_init} 
        self.step_fn = build_step_fn(robot)
        self.dyn_sys = robot

    def step(self, q, tau, F=None):
        step_args = {'tau':tau,
                     'mu':self.x['mu'],
                     'cov':self.x['cov'],
                     'q_meas':q}
        res = self.step_fn.call(step_args)
        self.x['mu'] = res['mu_next']
        self.x['cov'] = res['cov_next']
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
