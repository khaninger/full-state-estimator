import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        Somewhat humerously, this class is stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par, sym_par = {}):
        self.contacts = []
        self.vars = {}
        self.jit_options = {} #{'jit':True, "jit_options":{"compiler":"gcc"}}

        self.load_kin_dyn(par['urdf'], par['urdf_path'])
        self.build_fwd_kin()

        par.update(sym_par)
        self.add_contact(par['contact_1'])
        self.fric_model = par['fric_model']

        self.build_disc_dyn(par['h'], sym_par)
        self.build_output()

    def add_contact(self, contact_model):
        """ Add the contact_model to the robot """
        x_i = self.x_ee[0]+self.x_ee[1]@contact_model['pos']
        n_i = contact_model['stiff']/ca.norm_2(contact_model['stiff'])
        J_i = ca.jacobian(n_i.T@x_i, self.vars['q'])
        F_i = ca.DM(contact_model['stiff']).T@(x_i - contact_model['rest'])
        tau_i = J_i.T@F_i
        contact_i = ca.Function('contact', [self.vars['q']], [tau_i], ['q'], ['tau_i'])
        self.contacts.append(contact_i)

    def load_kin_dyn(self, urdf, urdf_path):
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('gripper'))
        
        self.nq = model.nq
        self.nx = 2*model.nq
        self.ny = model.nq  # eventually needs an udpate to include f/t 

    def build_fwd_kin(self):
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq)
        self.vars['xi'] = ca.vertcat(self.vars['q'], self.vars['dq'])
        self.vars['ddq']= ca.SX.sym('ddq', self.nq)
        self.vars['tau_err'] = ca.SX.sym('tau_err', self.nq)
        q = self.vars['q']
        dq = self.vars['dq']
        
        self.x_ee = self.fwd_kin(q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        #print("zeros {}".format(self.fwd_kin(np.array([0.0, 0.0, -np.pi/2, 0.0, 0.0, 0.0]))))
        #print("test  {}".format(self.fwd_kin(np.array([0.0, 0, -np.pi/2, np.pi/2, 0.0, 0.0]))))

        #x_ee = cpin.forwardKinematics(self.cmodel, self.cdata, q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        J = ca.jacobian(self.x_ee[0], q)
        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@dq # then reshape the result into the right shape
        #Jd = cpin.computeForwardKinematicsDerivatives(self.cmodel, self.cdata, q, dq)
        
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.djac = ca.Function('dot_jacobian',  [q, dq], [Jd])
        
        self.d_fwd_kin = ca.Function('dx', [q, dq], [J@dq], ['q', 'dq'], ['dx'])
        self.dd_fwd_kin = ca.Function('ddx', [q, dq, self.vars['ddq']],
                                      [Jd + J@self.vars['ddq']],
                                      ['q', 'dq', 'ddq'], ['ddx'])
        
    def build_disc_dyn(self, h, sym_par):
        q = self.vars['q']
        dq = self.vars['dq']
        tau_err = self.vars['tau_err']

        Minv = cpin.computeMinverse(self.cmodel, self.cdata, q)
        tau_i = self.get_contact_forces(q, dq)
        tau_f = self.get_fric_forces(dq)
                                
        ddq =  Minv@(tau_err+tau_i+tau_f)

        fn_dict = {'xi':self.vars['xi'], 'tau_err':tau_err}
        fn_dict.update(sym_par)
        dq_next= dq + h*ddq
        q_next = q + h*dq_next
        fn_dict['xi_next'] = ca.vertcat(q_next, dq_next)
            
        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau_err', *sym_par.keys()],
                                     ['xi_next'], self.jit_options)
        self.build_A(h)
    
    def build_A(self, h):
        """ Makes the linearized dynamic matrix A for semi-explicit integrator
            h: time step in seconds
        """
        q = self.vars['q']
        dq = self.vars['dq']
        fn_dict = {'xi':self.vars['xi'],
                   'tau_err':self.vars['tau_err']}

        res = self.disc_dyn.call(fn_dict)

        fn_dict['A'] = ca.jacobian(res['xi_next'], self.vars['xi'])
        self.A_fn = ca.Function('A', fn_dict,  
                                ['xi', 'tau_err'],['A'], self.jit_options)
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

    def get_contact_forces(self, q, dq):
        """ Returns the state-dependent external forces
            q: joint pose
            dq: joint velocity
        """
        F = 0
        for con in self.contacts:
            F += con(q)
        return F

    def get_fric_forces(self, dq):
        return -dq*self.fric_model['visc']
