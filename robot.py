import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        Somewhat humerously, this class is stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par, opt_par = {}, est_par = {}):
        print("Building robot with contact pars: {}".format(par['contact_1']))
        print("optimization pars: {}".format(opt_par))
        print("estimation pars:   {}".format(est_par))
        self.contacts = []
        self.contact_displacements = [] # signed distance functions
        self.contact_pts = []
        self.vars = {}
        self.jit_options = {} #{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-O3"]}}

        self.load_kin_dyn(par['urdf'], par['urdf_path'])
        self.build_vars(est_par)
        self.build_fwd_kin()

        # Add contact model
        par['contact_1'].update(opt_par)
        par['contact_1'].update(est_par)
        self.add_contact(par['contact_1'])
        self.fric_model = par['fric_model']

        self.build_disc_dyn(par['h'], opt_par, est_par)
        self.build_output()

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
            F_i = ca.DM(contact_model['stiff']).T@(x_i - contact_model['rest'])
        else:
            F_i = contact_model['stiff'].T@(x_i - contact_model['rest'])
        tau_i = J_i.T@F_i
        contact_i = ca.Function('contact', [self.vars['q']], [tau_i], ['q'], ['tau_i'])
        self.contacts.append(contact_i)

    def load_kin_dyn(self, urdf, urdf_path):
        model = pin.buildModelFromUrdf(urdf_path, verbose=True)
        data = model.createData()
        self.cmodel = cpin.Model(model, verbose = True)
        self.cdata = self.cmodel.createData()
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('link_6'))
        
        self.nq = model.nq

    def build_vars(self, est_pars):
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq)
        self.vars['est_pars'] = ca.vertcat(*est_pars.values())
        
        self.vars['xi'] = ca.vertcat(self.vars['q'], self.vars['dq'], self.vars['est_pars'])
        self.vars['ddq']= ca.SX.sym('ddq', self.nq)
        self.vars['tau_err'] = ca.SX.sym('tau_err', self.nq)
        
        self.np = 0
        for v in est_pars.values():
            self.np += v.size()[0]
        self.nx = 2*self.nq+self.np
        self.ny = self.nq
        
    def build_fwd_kin(self):
        q = self.vars['q']
        dq = self.vars['dq']
        
        #self.x_ee = self.fwd_kin(q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        #print("zeros {}".format(self.fwd_kin(np.array([0.0, 0.0, -np.pi/2, 0.0, 0.0, 0.0]))))
        #print("test  {}".format(self.fwd_kin(np.array([0.0, 0, -np.pi/2, np.pi/2, 0.0, 0.0]))))

        self.x_ee = cpin.forwardKinematics(self.cmodel, self.cdata, q, dq) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        print(self.x_ee)
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
        
    def build_disc_dyn(self, h, opt_par, est_par):
        q = self.vars['q']
        dq = self.vars['dq']
        tau_err = self.vars['tau_err']

        Minv = cpin.computeMinverse(self.cmodel, self.cdata, q)
        tau_i, disp, cont_pt = self.get_contact_forces(q, dq)
        tau_f = self.get_fric_forces(dq)

        ddq =  Minv@(tau_err+tau_i+tau_f)
        
        fn_dict = {'xi':self.vars['xi'], 'tau_err':tau_err,
                   'disp':disp, 'cont_pt':cont_pt}
        fn_dict.update(opt_par)
        
        dq_next= dq + h*ddq
        q_next = q + h*dq_next
        fn_dict['xi_next'] = ca.vertcat(q_next, dq_next, self.vars.get('est_pars', []))
        
        self.disc_dyn =  ca.Function('disc_dyn', fn_dict,
                                     ['xi', 'tau_err', *opt_par.keys()],
                                     ['xi_next', 'disp', 'cont_pt'], self.jit_options).expand()
        print(self.disc_dyn)
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
                                ['xi', 'tau_err'],['A'], self.jit_options).expand()
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
        disp = []
        for d in self.contact_displacements:
            disp.append(d(q))
        disp = ca.vertcat(*disp)
        cont_pt = []
        for p in self.contact_pts:
            cont_pt.append(p(q))
        cont_pt = ca.vertcat(*cont_pt)
        return F, disp, cont_pt

    def get_fric_forces(self, dq):
        return -dq*self.fric_model['visc']
