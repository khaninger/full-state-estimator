import casadi as ca
import numpy as np

class ekf():
    """ This defines an EKF observer """
    def __init__(self, dyn_sys, x0, cov0, proc_noise, meas_noise):
        self.dyn_sys = dyn_sys
        self.x = x0     # initial state
        self.cov = cov0 # initial covariance
        self.proc_noise = proc_noise
        self.meas_noise = meas_noise

    def step(self, q = None, tau_err = None, F = None):
        """ Steps the observer based on the input at time t and observation at time t """
        # Standad EKF update. See, e.g. pg. 51 in Thrun 'Probabilistic Robotics'
        x_next = self.dyn_sys.disc_dyn(self.x, tau_err)      # predict state and output at next time step
        y_next = self.dyn_sys.output(x_next)
        A, C = self.dyn_sys.get_linearized(self.x, tau_err)   # get the linearized dynamics and observation matrices
        cov_next = A.T@self.cov@A + self.proc_noise
        K = cov_next@C.T@ca.inv(C@cov_next@C.T + self.meas_noise) # calculate Kalman gain
        self.x = x_next + K@(q - y_next)
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-K@C)@cov_next
        return self.x

    def likelihood(self, obs):
        return NotImplemented
