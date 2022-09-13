import casadi as ca
import numpy as np
from robot import robot

class ekf():
    """ This defines an EKF observer """
    def __init__(self, urdf, urdf_path, h, q0, cov0, proc_noise, meas_noise, fric_model):
        self.dyn_sys = robot(urdf, urdf_path, h, fric_model)
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state

        self.cov = np.diag(cov0)  # initial covariance
        self.proc_noise = np.diag(np.concatenate((proc_noise['pos'],
                                                  proc_noise['vel'])))
        self.meas_noise = np.diag(meas_noise['pos'])

    def step(self, q, tau_err, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        self.x['tau_err'] = tau_err
        x_next = self.dyn_sys.disc_dyn.call(self.x)     # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(self.x)   # get the linearized dynamics and observation matrices
        cov_next = A@self.cov@(A.T) + self.proc_noise
        L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(L)):
           raise ValueError("Nans in the L matrix")
        x_corr = ca.vertcat(x_next['q_next'], x_next['dq_next']) + L@(q - x_next['q_next'])
        self.x['q'] = x_corr[:self.dyn_sys.nq]
        self.x['dq'] = x_corr[self.dyn_sys.nq:]
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-L@C)@cov_next
        return self.x

    def likelihood(self, obs):
        return NotImplemented
