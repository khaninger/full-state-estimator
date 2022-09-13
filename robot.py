import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        Somewhat humerously, this class is stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, urdf, urdf_path, h, fric_model):
        self.env_dyns = []
        self.vars = {}
        self.fric_model = fric_model
        self.load_kin_dyn(urdf, urdf_path)
        self.build_fwd_kin()
        self.build_disc_dyn(h)
        self.build_output()

    def add_env_dyn(self, env_dyn):
        """ Append the env_dyn to the robot """
        self.env_dyns.append(env_dyn)

    def load_kin_dyn(self, urdf, urdf_path):
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('link_6'))
        
        self.nq = model.nq
        self.nx = 2*model.nq
        self.ny = model.nq  # eventually needs an udpate to include f/t 

    def build_fwd_kin(self):
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq)
        self.vars['ddq']= ca.SX.sym('ddq', self.nq)
        self.vars['tau_err'] = ca.SX.sym('tau_err', self.nq)
        q = self.vars['q']
        dq = self.vars['dq']
        
        x_ee = self.fwd_kin(q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        print(x_ee)
        #x_ee = cpin.forwardKinematics(self.cmodel, self.cdata, q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        J = ca.jacobian(x_ee[0], q)
        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@dq # then reshape the result into the right shape
        #Jd = cpin.computeForwardKinematicsDerivatives(self.cmodel, self.cdata, q, dq)
        
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.djac = ca.Function('dot_jacobian',  [q, dq], [Jd])

        self.d_fwd_kin = ca.Function('dx', [q, dq], [J@dq], ['q', 'dq'], ['dx'])
        self.dd_fwd_kin = ca.Function('ddx', [q, dq, self.vars['ddq']],
                                      [Jd + J@self.vars['ddq']],
                                      ['q', 'dq', 'ddq'], ['ddx'])

    def get_ddq(self, q, dq, tau_err):
        """ Returns the expression for the joint acceleration
            q: joint positions
            dq: joint velocities
            tau_err: motor torque minus gravitational and coriolis forces
        """
        Minv = cpin.computeMinverse(self.cmodel, self.cdata, q)
        J = self.jac(q)
        F_s = self.get_state_forces(self.fwd_kin(q), J@dq)
        tau_f = self.get_fric_forces(dq)
        
        #Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        #Jd = Jd.reshape(J.shape)@dq                             # then reshape the result into the right shape
        Jd = self.djac(q, dq)
        P_s = self.get_acc_forces(self.fwd_kin(q))

        return Minv@(tau_err-J.T@(P_s@Jd+F_s)+tau_f)

    def build_disc_dyn(self, h):
        q = self.vars['q']
        dq = self.vars['dq']
        tau_err = self.vars['tau_err']
        ddq = self.get_ddq(q, dq, tau_err)

        fn_dict = {'q':q, 'dq':dq, 'ddq':ddq, 'tau_err':tau_err}
        fn_dict['dq_next']= dq + h*ddq
        fn_dict['q_next'] = q + h*fn_dict['dq_next']

        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['q', 'dq', 'tau_err'],
                                     ['q_next', 'dq_next', 'ddq'])
        self.build_A(h)
    
    def build_A(self, h):
        """ Makes the linearized dynamic matrix A for semi-explicit integrator
            h: time step in seconds
        """
        q = self.vars['q']
        dq = self.vars['dq']
        tau_err = self.vars['tau_err']
        fn_dict = {'q':q, 'dq':dq, 'tau_err':tau_err}
        x_next = self.disc_dyn.call(fn_dict)
        x_next_concat = ca.vertcat(x_next['q_next'], x_next['dq_next'])
        x_concat = ca.vertcat(q, dq)
        fn_dict['A'] = ca.jacobian(x_next_concat, x_concat)
        self.A_fn = ca.Function('A', fn_dict,  
                                ['q', 'dq', 'tau_err'],['A'])
        '''
        ddq = self.get_ddq(q, dq, tau_err)
        ddq_q = ca.jacobian(ddq, q)
        ddq_dq = ca.jacobian(ddq, dq)
        I = ca.DM.eye(self.nq)
        A = ca.vertcat(ca.horzcat(I + h*h*ddq_q, h*h*ddq_dq),
                       ca.horzcat(h*ddq_q,   I + h*ddq_dq))

        fn_dict = {'q': q, 'dq': dq, 'tau_err': tau_err, 'A': A}
        self.A_fn =  ca.Function('A', fn_dict,
                                 ['q', 'dq', 'tau_err'],['A'])
        '''
    def build_output(self):
        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))

    def get_tcp_motion(self, q, dq, ddq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        ddx = self.dd_fwd_kin(q, dq, ddq)
        return x, dx, ddx

    def get_linearized(self, state):
        return self.A_fn.call(state)['A'], self.C

    def get_state_forces(self, x, dx):
        """ Returns the state-dependent external forces
            x:  TCP pose
            dx: TCP velocity
        """
        F = 0
        for env in self.env_dyns:
            F += env.eval(x, dx)
        return F

    def get_fric_forces(self, dq):
        return -dq*self.fric_model['visc']
        
    def get_acc_forces(self, x):
        """ Returns the acceleration-dependent external forces
        """
        return 0
        
