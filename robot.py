import pinocchio as pin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        Somewhat humerously, this class is stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, urdf):
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('base_link'))
        self.inv_dyn  = ca.Function.deserialize(kindyn.rnea())
        self.mass_mat = ca.Function.deserialize(kindyn.ccrba())
        self.fwd_dyn  = kindyn.aba
        self.nq = kindyn.nq()
        self.nx = 2*kindyn.nq()
        
        self.q = ca.SX.sym('q', kindyn.nq())
        self.dq = ca.SX.sym('dq', kindyn.nq())
        
        x = self.fwd_kin(self.q)  # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        self.jac = ca.Function('Jacobian', [self.q], [ca.jacobian(x[0], self.q)], ['q'], ['Jac'])

        self.env_dyns = []

    def add_env_dyn(self, env_dyn):
        """ Append the env_dyn to the robot """
        self.env_dyns.append(env_dyn)
        
    def get_ddq(self, q, dq, tau_err):
        """ Returns the expression for the joint acceleration
            q: joint positions
            dq: joint velocities
            tau_err: motor torque minus gravitational and coriolis forces
        """
        M = self.mass_mat(q)
        J = self.jac(q)
        
        F_s = self.get_state_forces(self.fwd_kin(q), J@dq)

        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@dq                             # then reshape the result into the right shape
        P_s = self.get_acc_forces(self.fwd_kin(q))
        
        return ca.inv(M)@(tau_err-J.T@(P_s@Jd+F_s))
        
    def build_A(self, h):
        """ Makes the linearized dynamic matrix A for semi-explicit integrator
            h: time step in seconds
        """
        tau_err = ca.SX.sym('tau_err', self.nq)

        ddq = self.get_ddq(q, dq, tau_err)
        ddq_q = ca.jacobian(ddq, q)
        ddq_dq = ca.jacobian(ddq, dq)
        I = ca.DM.eye(self.nq)
        A = ca.vertcat(ca.horzcat(I + h*h*ddq_q, h*h*ddq_dq),
                       ca.horzcat(h*ddq_q,   I + h*ddq_dq))
        
        return ca.Function('A', [q, dq, tau_err], [A],
                           ['q', 'dq', 'tau_err'],['A'])     
        
    def get_state_forces(self, x, dx):
        """ Returns the state-dependent external forces
            x:  TCP pose
            dx: TCP velocity
        """
        F = 0
        for env in self.env_dyns:
            F += env.eval(x, dx)
        return F
        
    def get_acc_forces(self, x):
        """ Returns the acceleration-dependent external forces
        """
        return 0
        
