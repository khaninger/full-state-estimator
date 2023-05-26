import numpy as np
import casadi as ca

class Contact():
    def __init__(self):
        self.contact_info = {} # contact displacement/point/force per contact model
        self.torques = {}      # contact torque per contact model
        self.est_pars = {}     # symbolic parameters
        self.pars = {}         # all contact parameters
        self.np = 0            # dim of symbolic params

    def build_vars(self, par, est_pars):
        xi_init = []
        cov_init_vec = []
        proc_noise_vec = []
        
        for contact, est_pars, in est_pars.items():
            for est_par in est_pars:
                name = contact+"_"+est_par
                xi_init.append(par[name])
                par[name] = ca.SX.sym(name, 3)
                self.est_pars[name] = par[name]
                cov_init_vec.append(par['cov_init'][est_par])
                proc_noise_vec.append(par['proc_noise'][est_par])
            print('Adding {} model with pars:'.format(contact))
            print('  _pos   {}\n  _stiff {}\n  _rest  {}'.format(par[contact+'_pos'],
                                                          par[contact+'_stiff'],
                                                          par[contact+'_rest']))
        est_pars_vec = ca.vertcat(*self.est_pars.values())
        self.np = est_pars_vec.shape[0]
        return est_pars_vec, xi_init, cov_init_vec, proc_noise_vec

    def build_contact(self, par, q, x_ee, opt_pars={}):
        self.contacts = par['contact_models']
        for contact in self.contacts:
            self.pars[contact+'_pos']   = par[contact+'_pos']
            self.pars[contact+'_rest']  = par[contact+'_rest']
            self.pars[contact+'_stiff'] = par[contact+'_stiff']
        self.pars.update(self.est_pars)

        fn_dict = {'q':q}
        fn_dict.update(self.est_pars)
        fn_dict.update(opt_pars)
        res_dict = {}
        for contact in self.contacts:
            x_i = x_ee[0]+x_ee[1]@par[contact+'_'+'pos']
            print(x_i)
            disp_i = x_i - par[contact+'_rest']
            n_i = par[contact+'_stiff']/ca.norm_2(par[contact+'_stiff'])
            J_i = ca.jacobian(n_i.T@x_i, q)
            if isinstance(par[contact+'_stiff'], np.ndarray):
                F_i = -ca.DM(par[contact+'_stiff']).T@(x_i-par[contact+'_rest'])
            else:
                F_i = -par[contact+'_stiff'].T@(x_i-par[contact+'_rest'])
            res_dict[contact+'_pt'] = x_i
            res_dict[contact+'_disp'] = disp_i
            res_dict[contact+'_force'] = F_i
            self.torques[contact] = J_i.T@F_i
        
        fn_dict.update(res_dict)
        self.contact_info = ca.Function(contact+'_info', fn_dict,
                                        ['q', *self.est_pars.keys(), *opt_pars.keys()],
                                        [*res_dict.keys()]) 
            
    def get_contact_torque(self, q):
        tau = 0
        for contact in self.contacts:
            tau += self.torques[contact]
        return tau
            
    def get_statedict(self, q, dq, sym_pars_vec):
        d = {'q':q}
        st = 0
        for par in self.est_pars.keys():
            d[par] = sym_pars_vec[st:st+3]
            st += 3

        contact_info = self.contact_info.call(d)
        d.update(contact_info)
        force = 0
        for c in self.contacts:
            force += contact_info[c+'_force']
        d['f_ee_obs']  = force

        return d
    
