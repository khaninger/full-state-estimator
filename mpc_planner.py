from robot import RobotDict
import casadi as ca
import numpy as np
import colorednoise
from decision_vars import *


class MpcPlanner:
    def __init__(self, mpc_params, icem_params, path):
        self.mpc_params = mpc_params  # mpc parameters
        self.H = mpc_params['planning_horizon']  # number of mpc steps
        self.dt = mpc_params['dt']  # sampling time
        self.robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], {}).param_dict
        self.nx = self.robots['free-space'].nx
        self.nq = self.robots['free-space'].nq
        self.constraint_slack = mpc_params['constraint_slack']
        self.modes = self.robots.keys()
        self.disc_dyn_mpc = {mode: self.robots[mode].dyn_mpc.map(self.H, 'serial') for mode in self.modes}
        self.rollouts = {mode: self.robots[mode].create_rollout(self.H) for mode in self.modes}
        self.beta = icem_params['beta']
        self.num_samples = icem_params['num_samples']
        self.alpha = icem_params['alpha']
        self.dim_samples = icem_params['dim_samples']  # (nq,H)
        self.mu = np.zeros(self.dim_samples)
        self.std = np.zeros(self.dim_samples)
        self.u_min = icem_params['u_min']
        self.u_max = icem_params['u_max']
        self.options = yaml_load(path, 'ipopt_params.yaml')   # function for loading yaml files
        self.num_iter = icem_params['num_iter']
        self.num_elites = icem_params['elite_num']



    def iCEM_warmstart(self, params):
        mu = self.mu
        std = self.std
        for i in range(self.num_iter):
            samples_noise = colorednoise.powerlaw_psd_gaussian(self.beta, size=(self.num_samples, self.nq, self.H))
            samples = np.clip(samples_noise * self.std + self.mu, self.u_min, self.u_max)
            x0 = params['list_particles']  # list of tuples with joint states and sampled mode for every particle
            rollout = np.zeros((self.num_samples, self.nx, self.H))
            cost = np.zeros(self.num_samples)
            for j in range(self.num_samples):
                cost[j], rollout[j] = self.rollouts[x0[j][0]](x0[j][1], samples[j])

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

        self.args['p'] = ca.vertcat(*[params[el] for el in params.keys()])  # update parameters for the solver

        # warm start nlp with iCEM
        best_traj, best_input = self.iCEM_warmstart(params)
        vars.set_x0('xi', best_traj)
        vars.set_x0('tau', best_input)
        self.args['x0'] = vars.get_x0()

        sol = self.solver(**self.args)

        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.vars.set_results(sol['x'])
        self.x_traj = {m: self.vars['x_' + m] for m in self.modes}

        # update mean of sampling distribution
        self.mu = self.vars['tau']
        self.std = np.ones(self.dim_samples)  # re-initialize std to be ones at each time-step

    def build_solver(self, params):
        nx = self.nx
        nq = self.nq
        ty = ca.MX
        # initialize empty NLP
        J = {mode: 0.0 for mode in self.modes}
        g = []
        lbg = []
        ubg = []
        vars = {}

        # symbolic variables for parameters, these get assigned to numerical values in solve()
        params_sym = param_set(params, symb_type=ty.sym)

        # build decision variables
        for m in self.modes: vars['x_' + m] = np.zeros((nx, self.H))
        vars['u'] = np.zeros((nq, self.H))
        ub, lb = self.build_dec_var_constraints()
        # turn the decision variables into a decision_var object
        self.vars = decision_var_set(x0=vars, ub=ub, lb=lb, symb_type=ty.sym)
        for mode in self.modes:
            dynamics = self.disc_dyn_mpc[mode](x=self.vars['x_'+ mode], u=self.vars['u'])

            Xk_next = dynamics['xi_next']
            J[mode] += ca.sum2(dynamics['st_cost'])
            g += [ca.reshape(params_init['xi_t'] - self.vars['xi_' + mode][:, 0], nx, 1)]
            g += [ca.reshape(Xk_next[:, :] - self.vars['xi_' + mode][:, :], nx * self.H, 1)]
            lbg += [self.constraint_slack] * nx * self.H
            ubg += [-self.constraint_slack] * nx * self.H

        # calculate total objective
        J_total = 0.0
        J_u_total = self.mpc_params['R'] * ca.sumsqr(self.vars['u'])
        J_total = J_u_total
        for mode in self.modes:
            J_total += params_sym['belief_' + mode] * J[mode]  # expected value

        # set-up dictionary of arguments to solve
        w, lbw, ubw = self.vars.get_dec_vectors()
        w0 = self.vars.get_x0()
        self.args = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        prob = {'f': J_total, 'x': w, 'g': ca.vertcat(*g), 'p': params_sym.get_vector()}
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_dec_var_constraints(self):
        pass







