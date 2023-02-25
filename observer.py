import casadi as ca
import numpy as np
from robot import robot

class ekf():
    """ This defines an EKF observer """
    def __init__(self, xi_init, cov_init, proc_noise, meas_noise):
        self.xi = xi_init
        self.cov = cov
        self.proc_noise = proc_noise
        self.meas_noise = meas_noise

    def step(self, xi_next, A, C):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        
        A, C = self.robot.dyn.get_linearized(xi_next, tau)   # get the linearized dynamics and observation matrices
        cov_next = A@self.cov@(A.T) + self.proc_noise
        L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(L)):
           raise ValueError("Nans in the L matrix")
        self.xi_corr = (xi_next + L@(q - xi_next[:self.dyn_sys.nq])).full()
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-L@C)@cov_next

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
