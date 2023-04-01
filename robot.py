import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn



class robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        This class should be stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par, opt_par = {}, est_par = {}):
        print("Building robot with contact pars: {}".format(par['contact_1']))
        print("optimization pars: {}".format(opt_par))
        print("estimation pars:   {}".format(est_par))
        self.contacts = []
        self.contact_displacements = [] # signed distance functions
        self.contact_pts = []           # list of contact models
        self.vars = {}                  # dictionary of state
        self.jit_options = {}#{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}

        self.load_kin_dyn(par['urdf'], par['urdf_path'])
        self.build_vars(est_par)
        self.build_fwd_kin()

        # Add contact model
        par['contact_1'].update(opt_par)
        par['contact_1'].update(est_par)
        self.add_contact(par['contact_1'])
        self.fric_model = par['fric_model']

        self.build_disc_dyn(par['h'], opt_par, est_par)

    def add_contact(self, contact_model):
        """ Add the contact_model to the robot """
        x_i = self.x_ee[0]+self.x_ee[1]@contact_model['pos']
        cont_i = ca.Function('cont', [self.vars['q']], [x_i], ['q'], ['x_i'])
        self.contact_pts.append(cont_i)
        disp_i = ca.Function('disp',[self.vars['q']],
                             [x_i - contact_model['rest']], ['q'],['xd'])
        self.contact_displacements.append(disp_i)
        n_i = contact_model['stiff']/ca.norm_2(contact_model['stiff'])
        J_i = ca.jacobian(n_i.T@x_i, self.vars['q'])
        if isinstance(contact_model['stiff'], np.ndarray):
            F_i = -ca.DM(contact_model['stiff']).T@(x_i-contact_model['rest'])
        else:
            F_i = -contact_model['stiff'].T@(x_i-contact_model['rest'])
        tau_i = J_i.T@F_i
        self.contacts.append(tau_i)

    def get_full_statedict(self, xi):
        # returns a statedict with all the bonus things (contact point, forces, etc) from state
        
        
    def statedict_to_vec(self, d):
        # Takes dict arg
        
    def load_kin_dyn(self, urdf, urdf_path):
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('tool0'))
        self.nq = self.model.nq

    def build_vars(self, est_pars):
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq)
        self.vars['est_pars'] = ca.vertcat(*est_pars.values())

        self.vars['xi'] = ca.vertcat(self.vars['q'], self.vars['dq'], self.vars['est_pars'])
        
        self.vars['ddq']= ca.SX.sym('ddq', self.nq)
        self.vars['tau'] = ca.SX.sym('tau', self.nq)
        
        self.np = 0
        for v in est_pars.values():
            self.np += v.size()[0]
        self.nx = 2*self.nq+self.np
        self.ny = self.nq
        
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
        self.dd_fwd_kin = ca.Function('ddx', [q, dq, self.vars['ddq']],
                                      [Jd + J@self.vars['ddq']],
                                      ['q', 'dq', 'ddq'], ['ddx'])
    
    def build_disc_dyn(self, h, opt_par, est_par):
        q = self.vars['q']
        dq = self.vars['dq']
        tau = self.vars['tau']
        B = ca.diag(self.fric_model['visc'])
        tau_err = tau - cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)

        M = cpin.crba(self.cmodel, self.cdata, q)+ca.diag(np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25]))
        Mtilde_inv = ca.inv(M+h*B)
        semiimplicit = (ca.DM.eye(self.nq)+h*ca.inv(M)@ca.diag(self.fric_model['visc']))

        tau_i, disp, cont_pt = self.get_contact_forces(q, dq)
        
        # Old-fashioned dynamics
        ddq =  ca.inv(M)@(tau_err+tau_i)
        dq_next= ca.inv(semiimplicit)@(dq + h*ddq)
        q_next = q + h*dq_next
        xi_next = ca.vertcat(q_next, dq_next, self.vars.get('est_pars', []))

        fn_dict = {'xi':self.vars['xi'], 'xi_next': xi_next, 'tau':tau}
        fn_dict.update(opt_par)
        
        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau', *opt_par.keys()],
                                     ['xi_next'], self.jit_options).expand()
        
        fn_dict['A'] = ca.jacobian(fn_dict['xi_next'], self.vars['xi'])
        self.A = ca.Function('A', {k:fn_dict[k] for k in ('A', 'xi', 'tau')},  
                                ['xi', 'tau'],['A'], self.jit_options).expand()        
        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))

        # New dynamics
        delta = Mtilde_inv@(-B@dq + tau_err + tau_i)
        fn_dict = {'xi':self.vars['xi'],
                   'tau':tau}
        fn_dict.update(opt_par)

        dq_next= dq + h*delta
        q_next = q + h*dq_next
        fn_dict['xi_next'] = ca.vertcat(q_next, dq_next, self.vars.get('est_pars', []))
        self.vars['xi_next'] = fn_dict['xi_next']       
        self.disc_dyn_opt =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau', *opt_par.keys()],
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

        self.A_opt =  ca.Function('A', {'xi': self.vars['xi'], 'tau': tau, 'A': A},
                                 ['xi', 'tau'],['A'], self.jit_options).expand()

        self.C =  np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))
        
    def get_tcp_motion(self, q, dq, ddq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        ddx = self.dd_fwd_kin(q, dq, ddq)
        return x, dx, ddx

    def get_linearized(self, state):
        return self.A.call(state)['A'], self.C

    def get_linearized_opt(self, state):
        return self.A_opt.call(state)['A'], self.C

    def get_contact_forces(self, q, dq):
        """ Returns the state-dependent external forces
            q: joint pose
            dq: joint velocity
        """
        tau = 0
        for con in self.contacts:
            tau += con
        disp = []
        for d in self.contact_displacements:
            disp.append(d(q))
        disp = ca.vertcat(*disp)
        cont_pt = []
        for p in self.contact_pts:
            cont_pt.append(p(q))
        cont_pt = ca.vertcat(*cont_pt)
        return tau, disp, cont_pt
