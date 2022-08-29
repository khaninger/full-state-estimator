import casadi as ca
import numpy as np
from robot import robot

class ekf():
    """ This defines an EKF observer """
    def __init__(self, urdf_path, h, q0, cov0, proc_noise, meas_noise, fric_model):
        self.dyn_sys = robot(urdf_path, h, fric_model)
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state
        self.cov = cov0 # initial covariance
        self.proc_noise = proc_noise
        self.meas_noise = meas_noise


    def step(self, q, tau_err, F = None):
        """ Steps the observer based on the input at time t and observation at time t """
        self.x['tau_err'] = tau_err
        #print(tau_err)
        # Standad EKF update. See, e.g. pg. 51 in Thrun 'Probabilistic Robotics'
        print(self.dyn_sys.mass_mat(self.x['q']))
        #print(self.x)
        x_next = self.dyn_sys.disc_dyn.call(self.x)      # predict state and output at next time step
        #print('ddq: {}'.format(x_next['dq_next']))
        A, C = self.dyn_sys.get_linearized(self.x)   # get the linearized dynamics and observation matrices
        #print(A)
        cov_next = A.T@self.cov@A + self.proc_noise
        K = cov_next@C.T@ca.inv(C@cov_next@C.T + self.meas_noise) # calculate Kalman gain
        x_corr = ca.vertcat(x_next['q_next'], x_next['dq_next']) + K@(q - x_next['q_next'])
        self.x['q'] = x_corr[:self.dyn_sys.nq]
        self.x['dq'] = x_corr[self.dyn_sys.nq:]
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-K@C)@cov_next
        return self.x

    def likelihood(self, obs):
        return NotImplemented
