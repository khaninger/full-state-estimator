import numpy as np
import casadi as ca

class Contact():
    def __init__(self):
        self.contact_forces = [] # contact force models
        self.contact_disps = []  # signed distance functions
        self.contact_pts = []    # list of contact models
        self.est_pars = []
        self.np = 0

    def build_vars(par, est_par_dict):
        xi_init = []
        cov_init_vec = []
        proc_noise_vec = []
        
        for contact, est_pars in est_par_dict.items():
            for est_par in est_pars:
                name = contact+"_"+est_par 
                par[name] = ca.SX.sym(name, 3)
                self.est_pars.append(par[name])
                xi_init.append(par[name])
                cov_init_vec.append(par['cov_init'][est_par])
                proc_noise_vec.append(par['proc_noise'][est_par])

        self.est_pars = ca.vertcat(*self.est_pars)
        self.np = self.est_pars.shape[0]
        return self.est_pars, xi_init, cov_init_vec, proc_noise_vec

    def build_contact(self, pars, q, x_ee):
        for contact in pars['contact_models']:
            x_i = x_ee[0]+x_ee[1]@pars[contact+'_'+'pos']
            self.contact_pts.append(ca.Function(contact+'_pt', [q], [x_i], ['q'], ['x_i']))
            self.contact_disps.append( ca.Function(contact+'_disp',[self.vars['q']],
                                 [x_i - contact_model['rest']], ['q'],['xd']))
            n_i = par[contact+'_stiff']/ca.norm_2(par[contact+'_stiff'])
            J_i = ca.jacobian(n_i.T@x_i, self.vars['q'])
            if isinstance(par[contact+'_stiff'], np.ndarray):
                F_i = -ca.DM(par[contact+'_stiff']).T@(x_i-par[contact+'_rest'])
            else:
                F_i = -par[contact+'_stiff'].T@(x_i-par[contact+'_rest'])
            self.contact_forces.append(J_i.T@F_i)

    def get_contact_forces(self, q, dq):
        """ Returns the state-dependent external forces
            q: joint pose
            dq: joint velocity
        """
        tau = 0
        for con in self.contact_forces:
            tau += con
        disps = ca.vertcat(*[disp(q) for disp in self.contact_disps]) 
        cont_pts = ca.vertcat(*[contact_pt(q) for contact_pt in self.contact_pts])
        return tau, disps, cont_pts
