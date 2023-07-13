from robot import RobotDict
import casadi as ca
import numpy as np
import colorednoise
from decision_vars import *


class MpcPlanner:
    def __init__(self, mpc_params, icem_params, ipopt_options):
        self.mpc_params = mpc_params  # mpc parameters
        self.icem_params = icem_params
        self.H = self.mpc_params['H']  # number of mpc steps
        self.dt = self.mpc_params['dt']  # sampling time
        self.robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], {}).param_dict
        self.nx = self.robots['free-space'].nx
        self.nq = self.robots['free-space'].nq
        self.N_p = self.mpc_params['N_p']  # dimensions of impedance
        self.constraint_slack = self.mpc_params['constraint_slack']
        self.modes = self.robots.keys()
        self.disc_dyn_mpc = {mode: self.robots[mode].dyn_mpc for mode in self.modes}  # for constructing overall cost
        self.rollouts = {mode: self.robots[mode].create_rollout for mode in self.modes}  # for rolling out trajectories in CEM
        self.beta = self.icem_params['beta']
        self.num_samples = self.icem_params['num_samples']
        self.alpha = self.icem_params['alpha']
        self.dim_samples = (self.N_p, self.H)  # (nq,H)
        self.mu = np.zeros(self.dim_samples)
        self.std = np.zeros(self.dim_samples)
        self.u_min = self.icem_params['u_min']
        self.u_max = self.icem_params['u_max']
        self.options = ipopt_options
        self.num_iter = self.icem_params['num_iter']
        self.num_elites = self.icem_params['elite_num']



    def iCEM_warmstart(self, params):
        mu = self.mu
        std = self.std
        for i in range(self.num_iter):
            samples_noise = colorednoise.powerlaw_psd_gaussian(self.beta, size=(self.num_samples, self.N_p, self.H))
            samples = np.clip(samples_noise * self.std + self.mu, self.u_min, self.u_max)
            x0 = params['list_particles']  # list of tuples with joint states and sampled mode for every particle
            des_pose = params['des_pose']
            imp_stiff = params['imp_stiff']
            rollout = np.zeros((self.num_samples, self.nx, self.H))
            cost = np.zeros(self.num_samples)
            for j in range(self.num_samples):
                cost[j], rollout[j] = self.rollouts[x0[j][0]](x0[j][1], samples[j], imp_stiff, des_pose)

            elite_indexes = np.argsort(cost)[:self.num_elites]
            elite_samples = samples[elite_indexes]
            new_mu = np.mean(elite_samples, axis=0)
            new_std = np.std(elite_samples, axis=0)
            mu = self.alpha*mu + (1-self.alpha)*new_mu
            std = self.alpha * std + (1 - self.alpha) * new_std
            best_state_traj = rollout[elite_indexes[0]].reshape(self.nx, self.H)

        return best_state_traj, mu

    def solve(self, params):

        if not hasattr(self, "solver"):
            self.build_solver(params)

        self.args['p'] = self.pars.update(params)  # update parameters for the solver

        # warm start nlp with iCEM
        best_traj, best_input = self.iCEM_warmstart(params)
        self.vars.set_x0('xi', best_traj)
        self.vars.set_x0('imp_rest', best_input)
        self.args['x0'] = self.vars.get_x0()

        sol = self.solver(**self.args)

        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.vars.set_results(sol['x'])
        # update mean of sampling distribution
        self.mu = self.vars['imp_rest']
        self.std = np.ones(self.dim_samples)  # re-initialize std to be ones at each time-step

        return self.vars.filter()    # return dictionary of decision variables

    def build_solver(self, params0):
        nx = self.nx
        nq = self.nq
        ty = ca.MX
        N_p = self.N_p
        #opt_imp = self.mpc_params['opt_imp']

        # initialize empty NLP
        J = 0  # objective function
        vars0 = {}  # initial values for decision variables

        # symbolic variables for parameters, these get assigned to numerical values in solve()
        self.pars = param_set(params0)

        # build decision variables
        #if opt_imp: vars0['imp_stiff'] = self.pars['imp_stiff']   # imp stiff in tcp coord, initial value is current stiff
        vars0['imp_rest'] = np.zeros(N_p)
        for m in self.modes:
            vars0['q_' + m] = np.zeros((nx, self.H))  # joint trajectory relative to tcp
        ub, lb = self.build_dec_var_constraints()
        # turn the decision variables into a decision_var object
        self.vars = decision_var_set(x0=vars0, ub=ub, lb=lb, symb_type=ty.sym)
        imp_stiff = self.pars['imp_stiff']
        self.build_constraints()
        for mode in self.modes:
            dyn_next = self.disc_dyn_mpc[mode](xi=self.vars['q_' + mode],
                                               imp_rest=self.vars['imp_rest'],
                                               imp_stiff=imp_stiff,
                                               des_pose=self.pars['des_pose'])

            self.add_continuity_constraints(dyn_next['xi_next'], self.vars['q_' + mode])
            J += self.pars['belief_' + mode] * ca.sum2(dyn_next['cost'])

        # set up dictionary of arguments to solve
        x, lbx, ubx, x0 = self.vars.get_dec_vectors()

        self.args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        prob = dict(f=J, x=x, g=ca.vertcat(*self.g), p=self.pars.get_sym_vec())
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_constraints(self):
        # General NLP constraints, not including continuity constraints
        self.g = []  # constraints functions
        self.lbg = []  # lower bound on constraints
        self.ubg = []  # upper-bound on constraints
        self.g += [self.vars.get_deviation('imp_stiff')]
        self.lbg += [-self.mpc_params['delta_K_max']] * self.N_p
        self.ubg += [self.mpc_params['delta_K_max']] * self.N_p

    def add_continuity_constraints(self, x_next, x):
        nx = self.nx
        H = self.H
        self.g += [ca.reshape(self.pars['x0'] - x[:, 0], nx, 1)]
        self.g += [ca.reshape(x_next[:, :] - x[:, :], nx * H, 1)]
        self.lbg += [self.mpc_params['constraint_slack']] * nx * H
        self.ubg += [-self.mpc_params['constraint_slack']] * nx * H

    def build_dec_var_constraints(self):
        ub = {}
        lb = {}
        lb['des_pose'] = -self.mpc_params['delta_xd_max']
        ub['des_pose'] = self.mpc_params['delta_xd_max']
        lb['imp_stiff'] = self.mpc_params['K_min']
        ub['imp_stiff'] = self.mpc_params['K_max']

        return ub, lb














