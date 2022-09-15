import casadi as ca
import numpy as np
from robot import robot

class ekf():
    """ This defines an EKF observer """
    def __init__(self, par, q0, est_geom = False):
        self.x = {'q': q0,
                  'dq': ca.DM.zeros(q0.size)} # initial state

        if est_geom:
            est_par = {'contact_1_pose':ca.SX.sym('contact_1_pos',3)}
            self.x['p'] = ca.DM.zeros(3)
            self.dyn_sys = robot(par, est_par = est_par)
            self.proc_noise = np.diag(np.concatenate((par['proc_noise']['pos'],
                                                  par['proc_noise']['vel'],
                                                  par['proc_noise'].get('geom'))))
        else:
            self.dyn_sys = robot(par)
            self.proc_noise = np.diag(np.concatenate((par['proc_noise']['pos'],
                                                      par['proc_noise']['vel'])))
        self.cov = np.diag(par['cov_init'])  # initial covariance
        self.meas_noise = np.diag(par['meas_noise']['pos'])

    def step(self, q, tau_err, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau_err':tau_err,
                     'xi':ca.vertcat(self.x['q'], self.x['dq'], self.x.get('p'))}
        x_next = self.dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        cov_next = A@self.cov@(A.T) + self.proc_noise
        L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(L)):
           raise ValueError("Nans in the L matrix")
        #print(q-x_next['q_next'])
        xi_corr = x_next['xi_next'] + L@(q - x_next['xi_next'][:self.dyn_sys.nq])
        self.x['q'] = xi_corr[:self.dyn_sys.nq].full()
        self.x['dq'] = xi_corr[self.dyn_sys.nq:2*self.dyn_sys.nq].full()
        self.x['p'] = xi_corr[2*self.dyn_sys.nq:].full()
        self.x['xi'] = xi_corr.full()
        self.cov = (ca.DM.eye(self.dyn_sys.nx)-L@C)@cov_next
        return self.x

    def likelihood(self, obs):
        return NotImplemented
