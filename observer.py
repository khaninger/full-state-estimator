import casadi as ca
import numpy as np
from robot import Robot

def build_step_fn(self, robot):
    # Build a static KF update w arguments of mu, sigma, tau and measured q
    proc_noise = robot.proc_noise
    meas_noise = robot.meas_noise

    tau = robot.vars['tau']
    mu = robot.vars['xi']
    mu_next = robot.vars['xi_next']
    A, C = robot.get_linearized_opt({'tau':tau, 'xi':mu})

    q_meas = ca.SX.sym('q_meas', robot.nq)
    cov = ca.SX.sym('cov', mu.shape[0], mu.shape[0])

    cov_next = A@cov@(A.T) + proc_noise
    L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + meas_noise) # calculate Kalman gain

    mu_next_corr = mu_next + L@(q_meas - mu_next[:robot.nq])
    cov_next_corr = (ca.SX.eye(robot.nx)-L@C)@cov_next # corrected covariance

    fn_dict = {'tau':tau, 'mu':mu, 'cov':cov, 'q_meas':q_meas,
               'mu_next':mu_next_corr, 'cov_next':cov_next_corr}
    self.step_fn = ca.Function('ekf_step', fn_dict,
                               ['tau', 'mu', 'cov', 'q_meas'], # inputs to casadi function
                               ['mu_next', 'cov_next'])        # outputs of casadi function
    return step_fn

class ekf():
    """ This defines an EKF observer """
    def __init__(self, robot):
        self.x = {'mu':robot.xi_init, 'cov':robot.cov_init} 
        self.step_fn = build_step_fn(robot)

    
    def step_fast(self, q, tau, F=None):
        step_args = {'tau':tau,
                     'mu':self.x['mu'],
                     'cov':self.x['cov'],
                     'q_meas':q}
        res = self.step_fn.call(step_args)
        self.x['mu'] = res['mu_next'].full()
        self.x['cov'] = res['cov_next']
        return self.x

    def step(self, q, tau, dyn_sys, F = None):
        """ Steps the observer baed on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau':tau,
                     'xi':self.x['mu']}
        
        x_next = dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        A, C = dyn_sys.get_linearized(step_args)   # get the linearized dynamics and observation matrices
        cov_next = A@self.x['cov']@(A.T) + self.proc_noise
        print(cov_next)

        self.L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + self.meas_noise) # calculate Kalman gain
        if np.any(np.isnan(self.L)): raise ValueError("Nans in the L matrix")
    
        xi_corr = x_next['xi_next'] + self.L@(q - x_next['xi_next'][:dyn_sys.nq])
        self.x['xi'] = xi_corr.full()
        #self.x['q'] = xi_corr[:dyn_sys.nq].full()
        #self.x['dq'] = xi_corr[dyn_sys.nq:2*dyn_sys.nq].full()
        #if self.est_geom: self.x['p'] = xi_corr[2*dyn_sys.nq:].full()
        #if self.est_stiff: self.x['stiff'] = xi_corr[2*dyn_sys.nq:].full().flatten()
        self.x['cov'] = (ca.DM.eye(dyn_sys.nx)-self.L@C)@cov_next # corrected covariance
    
        
        #self.x['cont_pt'] = x_next['cont_pt'].full().flatten()
        #x_ee = self.dyn_sys.fwd_kin(self.x['q'])
        #self.x['x_ee'] = (x_ee[0].full(),
        #                  x_ee[1].full())
        #self.mom_obs.step(x_next['mom'], x_next['tau_err'])
        #self.x['f_ee_mo'] =  (self.dyn_sys.jacpinv(self.x['q'])@self.mom_obs.r).full()
        #self.x['f_ee_obs'] = -(self.dyn_sys.jacpinv(self.x['q'])@x_next['tau_i']).full()
        
        
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
