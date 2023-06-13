import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
import casadi as ca
import ruamel.yaml

from contact import Contact

class Robot():
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization
        This class should be stateless (e.g. the actual system state shouldn't be stored here).
    """
    def __init__(self, par, opt_pars = {}, est_pars = {}, flag=True):
        """ IN: par is the complete parameter dictionary
            IN: opt_par is a dict, containing the sym vars directly
            IN: est_par is a dict, key is contact name, value is list of params
            IN: flag is just for distinguishing between different observation matrices
        """
        print("Building robot model with:")
        print("  contact model(s):  {}".format(par['contact_models']))
        print("  optimization pars: {}".format(opt_pars))
        print("  estimation pars:   {}".format(est_pars))
    
        self.vars = {}       # dictionary of state as symbolic variables
        self.jit_options = {}#{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}

        self.contact = Contact()
        self.new_obsMatrix = flag
        self.load_kin(par['urdf_path'])
        self.build_vars(par, opt_pars, est_pars)

        self.fric_model = par['fric_model']

        self.build_disc_dyn(par['h'], opt_pars)
        self.disc_dyn_mpc(par['h'], opt_pars)


    def get_statedict(self, xi):
        # Maps from a vector xi to a state dictionary
        d = {'q':xi[:self.nq],
             'dq':xi[self.nq:2*self.nq],
             'xi':xi}
        d.update(self.contact.get_statedict(d['q'], d['dq'], xi[2*self.nq:]))
        return d
        
    def load_kin(self, urdf_path):
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.nq = self.model.nq

    def build_vars(self, par,       # initial parameters
                         opt_pars,  # parameters to be optimized
                         est_pars): # parameters to be estimated
        self.vars['q'] = ca.SX.sym('q', self.nq)
        self.vars['dq'] = ca.SX.sym('dq', self.nq) 
        self.vars['tau'] = ca.SX.sym('tau', self.nq)

        self.build_fwd_kin()
        par.update(opt_pars)
        self.vars['est_pars'], xii_con, ci_con, pn_con = self.contact.build_vars(par, est_pars)
        if par['contact_models'] != []:
            self.contact.build_contact(par, self.vars['q'], self.x_ee, opt_pars)
            
        xi_init = [par['q0'], ca.DM.zeros(self.nq), *xii_con]
        cov_init_vec = [par['cov_init']['q'], par['cov_init']['dq'], *ci_con]
        proc_noise_vec = [par['proc_noise']['q'], par['proc_noise']['dq'], *pn_con]

        self.vars['xi'] = ca.vertcat(self.vars['q'], self.vars['dq'], self.vars['est_pars'])
        self.xi_init = ca.vertcat(*xi_init)
        self.cov_init = ca.diag(ca.vertcat(*cov_init_vec))
        self.proc_noise = ca.diag(ca.vertcat(*proc_noise_vec))
        self.meas_noise = ca.diag(ca.vertcat(par['meas_noise']['pos'], par['meas_noise']['torques']))
        #self.meas_noise = ca.diag(ca.vertcat(par['meas_noise']['pos']))

        self.nx = 2*self.nq+self.contact.np
        self.ny = 2*self.nq  # with panda outputs are joint positions and torques
        
    def build_fwd_kin(self):
        q = self.vars['q']
        dq = self.vars['dq']
        ddq = ca.SX.sym('ddq', 7)
        cpin.forwardKinematics(self.cmodel, self.cdata, q, dq, ddq)
        cpin.updateFramePlacement(self.cmodel, self.cdata, self.cmodel.getFrameId('fr3_link8'))
        ee = self.cdata.oMf[self.cmodel.getFrameId('fr3_link8')]
        self.fwd_kin =  ca.Function('p',[q],[ee.translation, ee.rotation])
        #print(self.fwd_kin(np.array([-0.58, 0.54, -0.39, -1.83, 0.27, 2.35, -2.68])))
        self.x_ee = self.fwd_kin(q) # x is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
    
        J = ca.jacobian(self.x_ee[0], q)
        Jd = ca.jacobian(J.reshape((np.prod(J.shape),1)), q)@dq # Jacobian on a matrix is tricky so we make a vector
        Jd = Jd.reshape(J.shape)@dq # then reshape the result into the right shape
        
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv']) 
        self.djac = ca.Function('dot_jacobian',  [q, dq], [Jd])
        
        self.d_fwd_kin = ca.Function('dx', [q, dq], [J@dq], ['q', 'dq'], ['dx'])
    
    def build_disc_dyn(self, h, opt_pars):
        nq = self.nq
        nq2 = 2*self.nq
        q = self.vars['q']
        dq = self.vars['dq']
        #tau = self.vars['tau']
        B = ca.diag(self.fric_model['visc'])
        #tau_err = tau - cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q) # Difference between input torques and dynamics torques
        tau_err = ca.DM.zeros(self.nq)  # Torques from dynamic error. We assume the dynamics torques are compensated by the inner controller

        self.vars['tau_g'] = cpin.computeGeneralizedGravity(self.cmodel, self.cdata, self.vars['q'])  # gravitational torques
        #tau_err = -self.vars['tau_g']
        M = cpin.crba(self.cmodel, self.cdata, q)+ca.diag(0.5*np.ones(self.nq))
        Mtilde_inv = ca.inv(M+h*B)

        tau_i = self.contact.get_contact_torque(q)  # get total estimated contact torque
        self.vars['tau_i'] = tau_i  # make contact torque an independent variable

        delta = Mtilde_inv@(-B@dq + tau_err + tau_i)

        dyn_fn_dict = {'xi': self.vars['xi']}
        obs_fn_dict = {'xi': self.vars['xi'],
                       'tau': -self.vars['tau_i']}

        dyn_fn_dict.update(opt_pars)
        obs_fn_dict.update(opt_pars)

        self.obs = ca.Function('obs', obs_fn_dict, ['xi', *opt_pars.keys()], ['tau'])
        self.vars['y'] = ca.vertcat(q, obs_fn_dict['tau'])

        dq_next= dq + h*delta
        q_next = q + h*dq_next
        
        dyn_fn_dict['xi_next'] = ca.vertcat(q_next, dq_next, dyn_fn_dict['xi'][nq2:])
        self.vars['xi_next'] = dyn_fn_dict['xi_next']
        
        self.disc_dyn = ca.Function('disc_dyn', dyn_fn_dict,
                                     ['xi', *opt_pars.keys()],
                                     ['xi_next'], self.jit_options).expand()    
    
        ddelta_dq = Mtilde_inv@ca.jacobian(tau_i, q) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup
        ddelta_ddq = -Mtilde_inv@B
        ddelta_dp = Mtilde_inv@ca.jacobian(tau_i, self.vars['xi'][nq2:]) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup

        #A = ca.jacobian(self.vars['xi_next'], self.vars['xi']) # Old method which is slightly less efficient
        A = ca.SX.eye(self.nx)
        A[:nq, :nq] +=  h*h*ddelta_dq
        A[:nq, nq:nq2] += h*ca.SX.eye(nq)+h*h*ddelta_ddq
        A[:nq, nq2:] += h*h*ddelta_dp
        A[nq:nq2, :nq] += h*ddelta_dq
        A[nq:nq2, nq:nq2] += h*ddelta_ddq
        A[nq:nq2, nq2:] += h*ddelta_dp
        dyn_fn_dict['A'] = A
        self.A = ca.Function('A', {k:dyn_fn_dict[k] for k in ('A', 'xi', *opt_pars.keys())},
                                 ['xi', *opt_pars.keys()],['A'], self.jit_options).expand()
        #print(self.A)
        
        self.C_positions = np.hstack((np.eye(self.nq), np.zeros((self.nq, self.nx-self.nq))))   # previous constant observation matrix with only joint positions
        C_tau = ca.jacobian(-tau_i, self.vars['xi'])
        dyn_fn_dict['C'] = ca.vertcat(self.C_positions, C_tau)  # build new observation matrix with joint positions and torques
        self.C_torques = ca.Function('C', {k: dyn_fn_dict[k] for k in ('C', 'xi', *opt_pars.keys())},
                             ['xi', *opt_pars.keys()], ['C'], self.jit_options).expand()  # build new casadi function for new observation matrix
        #print(self.A.call({'xi': self.vars['xi']})['A'].shape)

    def disc_dyn_mpc(self, h, opt_pars):
        nq = self.nq
        nq2 = 2 * self.nq
        q = self.vars['q']
        dq = self.vars['dq']
        tau = self.vars['tau']  # input optimization variables
        B = ca.diag(self.fric_model['visc'])
        tau_err = tau - cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)  # Difference between input torques and dynamics torques
        M = cpin.crba(self.cmodel, self.cdata, q) + ca.diag(0.5 * np.ones(self.nq))
        Mtilde_inv = ca.inv(M + h * B)
        tau_i = self.contact.get_contact_torque(q)  # get total estimated contact torque
        delta = Mtilde_inv @ (-B @ dq + tau_err + tau_i)
        dyn_mpc_dict = {'xi': self.vars['xi'],
                        'tau': tau}
        dyn_mpc_dict.update(opt_pars)
        dq_next = dq + h * delta
        q_next = q + h * dq_next
        dyn_mpc_dict['xi_next'] = ca.vertcat(q_next, dq_next, dyn_mpc_dict['xi'][nq2:])
        self.dyn_mpc = ca.Function('disc_dyn', dyn_mpc_dict,
                                    ['xi', 'tau', *opt_pars.keys()],
                                    ['xi_next'], self.jit_options).expand()

    def create_rollout(self, H):
        input_samples = ca.SX.sym('input_samples', self.nq, H)
        xi = self.vars['xi']
        rollout_dict = {'xi': xi,
                        'input_samples': input_samples}
        rollout_fn = self.dyn_mpc.mapaccum(H)
        rollout = rollout_fn(self.vars['xi'], input_samples)

        cost = ca.sumsqr(input_samples)
        rollout_dict['cost'] = cost
        rollout_dict['best_state_traj'] = rollout
        roll_cost = ca.Function('roll_cost', rollout_dict,
                                ['xi', 'input_samples'],
                                ['cost', 'best_state_traj'])
        return roll_cost

    def get_tcp_motion(self, q, dq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        return x, dx

    def get_linearized(self, state):
        if self.new_obsMatrix:
            return self.A.call(state)['A'], self.C_torques.call(state)['C']
        else:
            return self.A.call(state)['A'], self.C_positions

class RobotDict():
    """
    This class creates the dictionary of robot instances according to different configuration files,
    associated with different dynamic models.

    """

    def __init__(self, robot_path=None, file_path=None, est_par={}):
        """
        file_path: list of file_paths associated to different contact models parameters.
        robot_path: path of robot configuration file

        """
        self.robot_param_dict = {}
        if robot_path:
            self.create_robot_dict(robot_path)
        self.param_dict = {}
        self.raw_param_dict = {}
        if file_path:
            self.load_robot_models(file_path, est_par)

    def create_robot_dict(self, robot_path):
        robot_file = open(robot_path, 'r')
        robot_content = ruamel.yaml.load(robot_file, Loader=ruamel.yaml.Loader)
        local_list = []
        for key, value in robot_content.items():
            local_list.append((key, value))
        self.robot_param_dict = dict(local_list)

    def yaml_load(self, path):
        yaml_file = open(path, 'r')
        yaml_content = ruamel.yaml.load(yaml_file, Loader=ruamel.yaml.Loader)
        local_list = []
        for key, value in yaml_content.items():
            if key.startswith('contact_1'):
                value = ca.DM(value)
            local_list.append((key, value))
        final_dict = dict(local_list)
        final_dict.update(self.robot_param_dict)
        model_name = final_dict['model']
        self.raw_param_dict[model_name] = final_dict
        return model_name, final_dict

    def load_robot_models(self, filepath, est_pars):
        dict_lst = []
        for i in range(len(filepath)):
            dict_lst.append((self.yaml_load(filepath[i])[0], self.yaml_load(filepath[i])[1]))
        save_dict = dict(dict_lst)
        for value in save_dict.values():
            key = value['model']
            print(f"loading model: {key}")
            model = Robot(value, est_pars)
            self.param_dict[key] = model

