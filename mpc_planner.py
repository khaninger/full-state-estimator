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
        self.disc_dyn = {mode: self.robots[mode].disc_dyn.map(self.H, 'serial') for mode in self.modes}
        self.disc_rollouts = {mode: self.robots[mode].disc_dyn.mapaccum(self.H) for mode in self.modes}
        self.beta = icem_params['beta']
        self.num_samples = icem_params['num_samples']
        self.alpha = icem_params['alpha']
        self.dim_samples = icem_params['dim_samples']
        self.mu = np.zeros(self.dim_samples)
        self.std = np.zeros(self.dim_samples)
        self.u_min = icem_params['u_min']
        self.u_max = icem_params['u_max']
        self.options = yaml_load(path, 'ipopt_params.yaml')
        self.cost_eval = build_cost_eval()  # TO DO: write casadi function for evaluating the stage cost


    def iCEM_warmstart(self, params):
        samples_noise = colorednoise.powerlaw_psd_gaussian(self.beta, size=(self.num_samples, self.nq, self.H))
        samples = np.clip(samples_noise*self.std + self.mu, self.u_min, self.u_max)
        x0 = params['']  # update initial setr of particles at each iteration --> should be a list of tuples
        rollout = np.zeros((self.num_samples, self.nx, self.H))
        cost = np.zeros((self.num_samples, 1))
        for i in range(self.num_samples):
            rollout[i] = self.disc_rollouts[x0[i][0]](x0[i][1], samples[i])
            cost[i] = self.cost_eval(rollout[i], samples[i])
        min_cost_index = np.argmin(cost, axis=0)
        best_state_traj = rollout[min_cost_index].reshape(self.nx, self.H)
        best_input_traj = samples[min_cost_index].reshape(self.nq, self.H)

        return best_state_traj, best_input_traj


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
        new_std = np.std(np.array((best_input, self.mu)), axis=0)
        self.std = self.std*self.alpha + (1-self.alpha)*new_std

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
            Fk_next = self.disc_dyn[mode](x=self.vars['x_'+ mode],
                                          u=self.vars['u'],
                                          init_pose=params_sym['init_pose'])

            Xk_next = Fk_next['xf']
            J[mode] += ca.sum2(Fk_next['st_cost'])
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







