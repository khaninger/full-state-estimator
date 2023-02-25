import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class Robot():
    '''
    This holds all configuration and contact models
    '''
    def __init__(self, q0, cov_init, contact_params, est_par = []):
        print("Building robot with contact pars: {}".format(par['contact_1']))
        print("estimation pars:   {}".format(est_par))

        # xi is the state which gets fed to the dynamics
        self.xi = {'q': q0,
                   'dq': ca.DM.zeros(q0.size)}
        
        for par in est_par: # getting initialized by what's in pars
            self.xi[par] = contact_params.get(par, np.zeros(3))
        
        self.contact = Contact(contact_params) 
        self.robot = RobotDynamic(par)
    
    def step(self, q, tau, F = None):
        xi = ca.vertcat(self.q, self.dq, self.contact.get_
        xi_next = self.robot.step(q = q, tau = tau, F = F)  # predict state and output at next time step
        
    def get_additional_state(self, xi):
        # Returns a dictionary for all the additional state
        #self.x['q'] = xi_corr[:self.dyn_sys.nq].full()
        #self.x['dq'] = xi_corr[self.dyn_sys.nq:2*self.dyn_sys.nq].full()
        #if self.est_geom: self.x['p'] = xi_corr[2*self.dyn_sys.nq:].full()
        #if self.est_stiff: self.x['stiff'] = xi_corr[2*self.dyn_sys.nq:].full().flatten()
        #self.x['cont_pt'] = x_next['cont_pt'].full().flatten()
        #x_ee = self.dyn_sys.fwd_kin(self.x['q'])
        #self.x['x_ee'] = (x_ee[0].full(),
        #                  x_ee[1].full())
        #self.x['xi'] = xi_corr.full()
        #self.x['f_ee_mo'] =  (self.dyn_sys.jacpinv(self.x['q'])@x_next['tau_err']).full()
        #self.x['f_ee_obs'] = (self.dyn_sys.jacpinv(self.x['q'])@x_next['tau_i']).full()

class Contact():
    '''
    This class holds contact parameters, which can be either symbolic,
    e.g. when being optimized or estimated, or numerical
    '''
    def __init__(self, params):
        self.pos = params.get(pos, ca.SX.sym('pos',3))
        self.stiff = params.get(stiff, ca.SX.sym('stiff',3))
        self.rest = params.get(rest, ca.SX.sym('rest',3))
        
        self.sym_params = [] # parameters which are symbolic, to be estimated
        if type(self.pos) == ca.SX: sym_params.append(self.pos)
        if type(self.stiff) == ca.SX: sym_params.append(self.stiff)
        if type(self.rest) == ca.SX: sym_params.append(self.rest)
        
    def get_force(self, x_ee):
        x_contact = self.x_ee[0]+self.x_ee[1]@self.pos
        disp = x_con - self.rest
        F = (self.stiff.T)@disp
        return F, disp, x_contact

    def get_joint_torque(self, q, x_ee):
        # Get the joint torque in q
        F, _, x_contact = self.get_force(x_ee)
        n_i = self.stiff/ca.norm_2(self.stiff)
        J_i = ca.jacobian(n_i.T@x_contact, q)
        return J_i.T@F
    
    def get_sym_params():
        return ca.vertcat(*self.sym_params)

class RobotDynamics():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        This class should be stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par):
        self.jit_options = {} #{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-O3"]}}
        self.load_kin_dyn(par['urdf'], par['urdf_path'])

        self.q = ca.SX.sym('q', self.nq)
        self.dq = ca.SX.sym('dq', self.nq)
        self.tau = ca.SX.sym('tau', self.nq)
        
        self.build_fwd_kin()

        self.fric_model = par['fric_model']

        self.build_disc_dyn(par['h'], opt_par, est_par)

    def load_kin_dyn(self, urdf, urdf_path):
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('tool0'))
        self.nq = self.model.nq
                       
    def build_fwd_kin(self):
        self.x_ee = self.fwd_kin(self.q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
    
        J = ca.jacobian(self.x_ee[0], self.q)
        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), self.q)@self.dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@self.dq # then reshape the result into the right shape
        
        self.jac = ca.Function('jacobian', [self.q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [self.q], [ca.pinv(J.T)], ['q'], ['pinv']) 
        self.djac = ca.Function('dot_jacobian',  [self.q, self.dq], [Jd])
        
        self.d_fwd_kin = ca.Function('dx', [self.q, self.dq], [J@self.dq], ['q', 'dq'], ['dx'])
        self.dd_fwd_kin = ca.Function('ddx', [self.q, self.dq, self.vars['ddq']],
                                      [Jd + J@self.vars['ddq']],
                                      ['q', 'dq', 'ddq'], ['ddx'])
    
    def build_disc_dyn(self, h, contact):
        # Build the dynamics, which maps
        # (q, dq, symbolic_params) -> (q_next, dq_next)
        # where symbolic_params are the symbolic parameters of the env primitives
        Minv = cpin.computeMinverse(self.cmodel, self.cdata, self.q)
        
        tau_err = self.tau - cpin.computeGeneralizedGravity(self.cmodel, self.cdata, self.q)
        tau_contact = contact.get_torque(self.q, self.x_ee)
        tau_f = -self.dq*self.fric_model['visc']
        
        ddq =  Minv@(tau_err+tau_contact+tau_f)
                        
        dq_next = dq + h*ddq
        q_next  = q + h*dq_next

        fn_dict = {'q': self.q, 'dq': self.dq, 'tau':self.tau,
                   'q_next':q_next, 'dq_next':dq_next}

        contact_params = contact.get_sym_params()
        fn_dict.update(contact_params)
        
        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['q', 'dq', 'tau', *contact_params.keys()],
                                     ['q_next', 'dq_next'], self.jit_options).expand()
    
    def build_lin_matrices(self, h, contact):
        """ Makes the linearized dynamic matrix A for semi-explicit integrator
            h: time step in seconds
        """
        fn_dict = {'q': self.q, 'dq': self.dq, 'tau':self.tau}
        est_params = contact.get_sym_params()
        num_est_params = est_params.shape[0]
        fn_dict.update(est_params)
        res = self.disc_dyn.call(fn_dict)
        
        xi = ca.vertcat(self.q, self.dq, *est_params)
        xi_next = ca.vertcat(res['q_next'], res['dq_next'], ca.SX(num_est_params))
        
        fn_dict['A'] = ca.jacobian(xi_next, xi)
        
        self.A_fn = ca.Function('A', fn_dict,  
                                ['xi', 'tau'], ['A'], self.jit_options).expand()
        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nq+num_est_params))))

    def get_tcp_motion(self, q, dq, ddq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        ddx = self.dd_fwd_kin(q, dq, ddq)
        return x, dx, ddx

    def get_linearized(self, state):
        return self.A_fn.call(state)['A'], self.C

