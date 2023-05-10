from robot import RobotDict
from decision_vars import *
import casadi as ca
import time
import numpy as np

H = 5     # num MPC steps
R = 0.1   # control weight in cost function
dt = 0.01 # time step in sec for discretized dynamics

solver_opts = {"print_time":False,
               "ipopt.print_level":0,
               "ipopt.tol":1.0e-5,
               "ipopt.acceptable_constr_viol_tol":2.0e-04,
               "ipopt.linear_solver":'MUMPS',
               "ipopt.warm_start_init_point":"yes",
               "ipopt.warm_start_bound_frac":1.0e-09,
               "ipopt.warm_start_bound_push":1.0e-09,
              }

robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], {})
robot = robots.param_dict['contact']

robot.build_disc_dyn(dt, {}) # Rebuild dyn with larger step size

nq = robot.nq
nx = 2*robot.nq
qd = np.ones((nq,1))

# Make the parameters, which are fixed numerical values which can be updated in each solve
params_init = dict(xi_t = np.zeros((2*robot.nq, 1)))
params = param_set(params_init, symb_type = ca.SX.sym)

# Make the decision variables, which are optimized
vars_init =  dict(tau = np.zeros((nq, H)),
                  xi = np.zeros((nx, H-1)))
vars_lb = dict(tau = -np.ones((nq, H)))
vars_ub = {k:-v for k,v in vars_lb.items()}
vars = decision_var_set(x0 = vars_init, lb = vars_lb, ub = vars_ub)

g = [] # lists for the constraints and it's bounds
J = 0  # objective function

dyn = robot.disc_dyn

#xi = ca.horzcat(params_init['xi_t'], vars['xi']) # using parameter for initial state
xi = ca.horzcat(np.zeros((nx,1)), vars['xi']) # some issue with parameters, just setting initial state to zero
xi_next = dyn(xi, vars['tau'])
print(xi_next)
g += [ca.reshape(xi_next[:,:-1]-vars['xi'][:,:], nx*(H-1), 1)]
J += ca.sumsqr(qd-xi_next[:nq,:])
J += ca.sumsqr(xi_next[nq:,:])
J += ca.sumsqr(R*vars['tau'])

x, x_lb, x_ub = vars.get_dec_vectors()
x0 = vars.get_x0()

args = dict(x0=x0, lbx=x_lb, ubx=x_ub)
prob = dict(f=J, x=x, g=ca.vertcat(*g))#, p=params.get_vector())
solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)



N = 100
times = []

xi_init = 0.01 * ca.DM.ones((nx, 1))

for i in range(N):
    # args['p'] = xi_init #+ 0.01*np.random.randn(nx)
    tic = time.perf_counter()
    sol = solver(**args)
    times.append(time.perf_counter() - tic)

    # Save solution + lagrangian for warm start
    args['x0'] = sol['x']
    args['lam_x0'] = sol['lam_x']
    args['lam_g0'] = sol['lam_g']
    vars.set_results(sol['x'])
    # print(vars['tau'])
print(f'Cold start time:  {times[0]} sec')
print(f'Cold start rate:  {1 / times[0]} Hz')
print(f'Warm start rate: {1 / np.mean(times[1:])} Hz')
