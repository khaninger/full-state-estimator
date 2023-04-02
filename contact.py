import numpy as np
import casadi as ca

class Contact():
    def __init__(self):
        self.forces = {} # contact force models
        self.torques = {} # contact force models
        self.disps = {}  # signed distance functions
        self.pts = {}    # list of contact models
        self.est_pars = []
        self.pars = {}
        self.np = 0

    def build_vars(self, par, est_par_dict):
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
                print('Adding {} model with {}:{}'.format(contact, est_par, par[name]))

        est_pars_vec = ca.vertcat(self.est_pars)
        self.np = est_pars_vec.shape[0]
        return est_pars_vec, xi_init, cov_init_vec, proc_noise_vec

    def build_contact(self, par, q, x_ee):
        self.contacts = par['contact_models']
        self.pars = {}
        for contact in self.contacts:
            self.pars[contact+'_pos']   = par[contact+'_pos']
            self.pars[contact+'_rest']  = par[contact+'_rest']
            self.pars[contact+'_stiff'] = par[contact+'_stiff']
            
            x_i = x_ee[0]+x_ee[1]@par[contact+'_'+'pos']
            self.pts[contact] = ca.Function(contact+'_pt', [q], [x_i], ['q'], ['x_i'])
            self.disps[contact] = ca.Function(contact+'_disp',[q],
                                                      [x_i - par[contact+'_rest']], ['q'],['xd'])
            n_i = par[contact+'_stiff']/ca.norm_2(par[contact+'_stiff'])
            J_i = ca.jacobian(n_i.T@x_i, q)
            if isinstance(par[contact+'_stiff'], np.ndarray):
                F_i = -ca.DM(par[contact+'_stiff']).T@(x_i-par[contact+'_rest'])
            else:
                F_i = -par[contact+'_stiff'].T@(x_i-par[contact+'_rest'])
            self.forces[contact] = ca.Function(contact+'_force', [q], [x_i], ['q'], ['x_i']) 
            self.torques[contact] = J_i.T@F_i

    def get_contact_torque(self, q):
        tau = 0
        for contact in self.contacts:
            tau += self.torques[contact]
        return tau
            
    def get_contact_forces(self, q, dq):
        """ Returns the state-dependent external forces
        """
        forces = {c+'_force':self.forces[c](q) for c in self.contacts}
        disps = {c+'_disp':self.disps[c](q) for c in self.contacts}
        pts =  {c+'_pt':self.pts[c](q) for c in self.contacts}
        return forces, disps, pts
    
    def get_statedict(self, q, dq, est_pars):
        forces, disps, pts = self.get_contact_forces(q, dq)
        force = 0
        for f in forces.values():
            force += f
        d = {'f_ee_obs':force}
        d.update(forces)
        d.update(disps)
        d.update(pts)
        d.update(self.pars)

        st = 0
        for par in self.est_pars:
            d[par] = xi[st:st+3]
            st += 3
            
        return d
    
