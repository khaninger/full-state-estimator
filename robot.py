import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

from contact import Contact

class Robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        This class should be stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par, opt_pars = {}, est_pars = {}):
        """ IN: par is the complete parameter dictionary
            IN: opt_par is a dict, containing the sym vars directly
            IN: est_par is a dict, key is contact name, value is list of params
        """
        print("Building robot model with")
        print("  optimization pars: {}".format(opt_pars))
        print("  estimation pars:   {}".format(est_pars))
    
        self.vars = {}       # dictionary of state as symbolic variables
        self.jit_options = {}#{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}

        self.contact = Contact()
        
        self.load_kin_dyn(par['urdf'], par['urdf_path'])
        self.build_vars(par, opt_pars, est_pars)
        self.build_fwd_kin()
        self.fric_model = par['fric_model']
                        
        self.contact.build_contact(par, self.vars['q'], self.x_ee)
        self.build_disc_dyn(par['h'], opt_pars)

    def get_statedict(self, xi):
        # Maps from a vector xi to a state dictionary
        d = {'q':xi[:self.nq],
             'dq':xi[self.nq:2*self.nq],
             'xi':xi}
        d.update(self.contact.get_statedict(d['q'], d['dq'], xi[2*self.nq:]))
        return d
        
    def load_kin_dyn(self, urdf, urdf_path):
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('tool0'))
        self.nq = self.model.nq

    def build_vars(self, par, opt_pars, est_pars):
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq)        
        self.vars['tau'] = ca.SX.sym('tau', self.nq)

        self.vars['est_pars'], xii_con, ci_con, pn_con = self.contact.build_vars(par, est_pars)
        
        xi_init = [par['q0'], ca.DM.zeros(self.nq), *xii_con]
        cov_init_vec = [par['cov_init']['pos'], par['cov_init']['vel'], *ci_con]
        proc_noise_vec = [par['proc_noise']['pos'], par['proc_noise']['vel'], *pn_con]

        self.vars['xi'] = ca.vertcat(self.vars['q'], self.vars['dq'], self.vars['est_pars'])
        self.xi_init = ca.vertcat(*xi_init)
        self.cov_init = ca.diag(ca.vertcat(*cov_init_vec))
        self.proc_noise = ca.diag(ca.vertcat(*proc_noise_vec))
        self.meas_noise = ca.diag(par['meas_noise']['pos'])
        
        self.nx = 2*self.nq+self.contact.np
        self.ny = self.nq

        par.update(opt_pars)
        
    def build_fwd_kin(self):
        q = self.vars['q']
        dq = self.vars['dq']
        
        self.x_ee = self.fwd_kin(q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
    
        J = ca.jacobian(self.x_ee[0], q)
        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@dq # then reshape the result into the right shape
        
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv']) 
        self.djac = ca.Function('dot_jacobian',  [q, dq], [Jd])
        
        self.d_fwd_kin = ca.Function('dx', [q, dq], [J@dq], ['q', 'dq'], ['dx'])
    
    def build_disc_dyn(self, h, opt_pars):
        q = self.vars['q']
        dq = self.vars['dq']
        tau = self.vars['tau']
        B = ca.diag(self.fric_model['visc'])
        tau_err = tau - cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)

        M = cpin.crba(self.cmodel, self.cdata, q)+ca.diag(np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25]))
        Mtilde_inv = ca.inv(M+h*B)
        semiimplicit = (ca.DM.eye(self.nq)+h*ca.inv(M)@ca.diag(self.fric_model['visc']))

        tau_i = self.contact.get_contact_torque(q)
        
        # Old-fashioned dynamics
        ddq =  ca.inv(M)@(tau_err+tau_i)
        dq_next= ca.inv(semiimplicit)@(dq + h*ddq)
        q_next = q + h*dq_next
        xi_next = ca.vertcat(q_next, dq_next, self.vars.get('est_pars', []))

        fn_dict = {'xi':self.vars['xi'], 'xi_next': xi_next, 'tau':tau}
        fn_dict.update(opt_pars)
        
        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau', *opt_pars.keys()],
                                     ['xi_next'], self.jit_options).expand()
        
        fn_dict['A'] = ca.jacobian(fn_dict['xi_next'], self.vars['xi'])
        self.A = ca.Function('A', {k:fn_dict[k] for k in ('A', 'xi', 'tau', *opt_pars.keys())},  
                                ['xi', 'tau', *opt_pars.keys()],['A'], self.jit_options).expand()        
        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))

        # New dynamics
        delta = Mtilde_inv@(-B@dq + tau_err + tau_i)
        fn_dict = {'xi':self.vars['xi'],
                   'tau':tau}
        fn_dict.update(opt_pars)

        dq_next= dq + h*delta
        q_next = q + h*dq_next
        fn_dict['xi_next'] = ca.vertcat(q_next, dq_next, self.vars.get('est_pars', []))
        self.vars['xi_next'] = fn_dict['xi_next']       
        self.disc_dyn_opt =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau', *opt_pars.keys()],
                                     ['xi_next'], self.jit_options).expand()    
        
        nq = self.nq
        nq2 = 2*self.nq

        ddelta_dq = Mtilde_inv@ca.jacobian(tau_i, q) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup
        ddelta_ddq = -Mtilde_inv@B
        ddelta_dp = Mtilde_inv@ca.jacobian(tau_i, self.vars['xi'][nq2:]) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup

        A = ca.SX.eye(self.nx)
        A[:nq, :nq] +=  h*h*ddelta_dq
        A[:nq, nq:nq2] += h*ca.SX.eye(nq)+h*h*ddelta_ddq
        A[:nq, nq2:] += h*h*ddelta_dp
        A[nq:nq2, :nq] += h*ddelta_dq
        A[nq:nq2, nq:nq2] += h*ddelta_ddq

        fn_dict['A_opt'] = A
        self.A_opt =  ca.Function('A', {k:fn_dict[k] for k in ('A_opt', 'xi', 'tau', *opt_pars.keys())},
                                 ['xi', 'tau',  *opt_pars.keys()],['A_opt'], self.jit_options).expand()

        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))
        
    def get_tcp_motion(self, q, dq, ddq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        return x, dx

    def get_linearized(self, state):
        return self.A.call(state)['A'], self.C

    def get_linearized_opt(self, state):
        return self.A_opt.call(state)['A'], self.C

