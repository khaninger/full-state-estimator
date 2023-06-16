import casadi as ca
import numpy as np
from robot import Robot
from scipy.stats import multivariate_normal
import copy

def build_step_fn(robot):
    # Build a static KF update w arguments of mu, sigma, tau and measured q
    proc_noise = robot.proc_noise
    meas_noise = robot.meas_noise

    #tau = robot.vars['tau'] - grav_torques
    
    mu = robot.vars['xi']
    mu_next = robot.vars['xi_next']
    tau_i = robot.vars['tau_i']  # expected contact torque
    tau_g = robot.vars['tau_g']  # gravitational torques
    y = robot.vars['y']          # expected pos and torque observations 

    A, C = robot.get_linearized({'xi': mu})  # get linearized state and observation matrices wrt states
    N = robot.nq
    q_meas = ca.SX.sym('q_meas', robot.nq)  # joint positions measurements
    tau_meas = ca.SX.sym('tau_meas', robot.nq)  # joint torques measurements
    y_meas = ca.vertcat(q_meas, tau_meas)  # stacked vector of measurements
    cov = ca.SX.sym('cov', mu.shape[0], mu.shape[0])

    cov_next = A@cov@(A.T) + proc_noise


    L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + meas_noise)  # calculate Kalman gain

    #print(L.shape)
    #y_hat = C@mu
    S_hat = C@cov_next@(C.T) + meas_noise
    #log_likelihood = -0.5*(np.log(ca.det(S_hat)) + ca.transpose(y_meas-y)@ca.inv(S_hat)@(y_meas-y) + N*np.log(2*np.pi))  # expression for log-likelihood
    #print(S_hat)
    #print(log_likelihood)
    [Q, R] = ca.qr(S_hat)  # QR decomposition for evaluating determinant efficiently
    det_S_t = ca.trace(R)  # determinant of original predicted measurement covariance is just the product of diagonal elements of R --> block triangular
    log_likelihood = -0.5 * (np.log(det_S_t) + ca.transpose(y_meas - y) @ ca.inv(S_hat) @ (y_meas - y) + N * np.log(2 * np.pi))
    temp1 = det_S_t**(-1/2)
    #temp1 = 0.1
    temp2 = ca.exp(-0.5*ca.transpose(y_meas-y) @ ca.inv(S_hat) @ (y_meas-y))
    mu_next_corr = mu_next + L@(y_meas - y)
    cov_next_corr = (ca.SX.eye(robot.nx)-L@C)@cov_next # corrected covariance
    likelihood = (2*np.pi)**(-N/2)*temp1*temp2
    #print(likelihood.shape)
    fn_dict = {'tau':tau_meas, 'mu':mu, 'cov':cov, 'q_meas':q_meas,
               'mu_next':mu_next_corr, 'cov_next':cov_next_corr, 'y_hat': y, 'S_hat': S_hat,
               'likelihood': likelihood, 'A': A, 'C': C, 'cov_next_pre': cov_next, 'Q': Q, 'tau_ext': tau_i, 'tau_g': tau_g, 'y_meas': y_meas}
    step_fn = ca.Function('ekf_step', fn_dict,
                          ['tau', 'mu', 'cov', 'q_meas'], # inputs to casadi function
                          ['mu_next', 'cov_next', 'S_hat', 'y_hat', 'likelihood', 'A', 'C', 'cov_next_pre', 'Q', 'tau_ext', 'tau_g', 'y_meas'])   # outputs of casadi function

    #print(step_fn)
    return step_fn

class ekf():
    """ This defines an EKF observer """
    def __init__(self, robot):
        self.x = {'mu':robot.xi_init, 'cov':robot.cov_init} 
        self.step_fn = build_step_fn(robot)
        self.dyn_sys = robot
        self.nq = robot.nq

    def step(self, q, tau, F=None):
        step_args = {'tau':tau,
                     'mu':self.x['mu'],
                     'cov':self.x['cov'],
                     'q_meas':q}
        res = self.step_fn.call(step_args)
        self.x['mu'] = res['mu_next']
        self.x['cov'] = res['cov_next']
        self.x['est_force'] = res['y_hat'][-self.nq:]
        self.x['tau_ext'] = res['tau_ext']
        self.x['tau_g'] = res['tau_g']
        self.x['y_meas'] = res['y_meas']
        return self.x

    def get_statedict(self):
        return self.dyn_sys.get_statedict(self.x['mu'])
    
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
