import pinocchio as pin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    def __init__(self, urdf):
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('base_link'))
        self.inv_dyn  = ca.Function.deserialize(kindyn.rnea())
        self.mass_mat = ca.Function.deserialize(kindyn.ccrba())
        self.fwd_dyn  = kindyn.aba
        self.nq = kindyn.nq()
        
        self.q = ca.SX.sym('q', kindyn.nq())
        x = self.fwd_kin(self.q)
        self.jac = ca.Function('Jacobian', [self.q], [ca.jacobian(x[0], self.q)], ['q'], ['Jac'])

        self.env_dyns = []

    def add_env_dyn(self, env_dyn):
        self.env_dyns.append(env_dyn)
        
    def get_ddq(self, q, dq, tau_err):      
        M = self.mass_mat(q)
       
        J = self.jac(q)
        F_s = self.get_state_forces(self.fwd_kin(q), J@dq)

        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq
        Jd = Jd.reshape(J.shape)@dq
        P_s = self.get_acc_forces(self.fwd_kin(q))
        
        return ca.inv(M)@(tau_err-J.T@(P_s@Jd+F_s))
        
    def build_A(self, h):
        q = ca.SX.sym('q', self.nq)
        dq = ca.SX.sym('dq', self.nq)
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
        F = 0
        for env in self.env_dyns:
            F += env.eval(x, dx)
        return F
        
    def get_acc_forces(self, x):
        return 0
        
class env_dyn():
    def __init__(self):
        self.params = {}
        self.state = {}

    def eval(self, x, dx):
        return 0
