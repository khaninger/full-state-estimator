import casadi as ca

class ekf():
    """ This defines an EKF observer """
    def __init__(self, dyn_sys, x0, cov0):
        self.dyn_sys = dyn_sys
        self.x = x0     # initial state
        self.cov = cov0 # initial covariance

    def step(self, inp, obs):
        """ Steps the observer based on the input at time t and observation at time t """
        # See, e.g. pg. 51 in Thrun 'Probabilistic Robotics'
        
        x_next, y_next = self.dyn_sys.step(inp)      # predict state and output at next time step
        A, C = self.dyn_sys.get_linearized(self.x)   # get the linearized dynamics and observation matrices
        R, Q = self.dyn_sys.get_noise(self.x)        # get the input (R) and observation (Q) noise matricesss
        cov_next = A.T@self.cov@A + R
        K = cov_next@C.T@ca.inv(C@cov_next@C.T + Q)  # calculate Kalman gain
        self.x = x_next + K@(obs - y_next)
        self.cov = (ca.eye(self.dyn_sys.nx)-K@C)@cov_next
        
    def likelihood(self, obs):
        return NotImplemented
