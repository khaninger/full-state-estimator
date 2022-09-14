import numpy as np
import casadi as ca
import time

def loss_fn(obs, state, param):

def get_dec_vectors(param):
    x = x0 = lbx = ubx = []
    for k in param.get_keys():
        x += [param[k]]
        if k == 'stiff':
            x0 += [ca.DM.zeros(3)]
            lbx += [ca.DM.zeros(3)]
            ubx += [1e6*ca.DM.ones(3)]
        if k == 'pos':
            x0 += [ca.DM.zeros(3)]
            lbx += [-0.2*ca.DM.ones(3)]
            ubx += [0.2*ca.DM.ones(3)]
        if k == 'rest_pos':
            x0 += [ca.DM.zeros(3)]
            lbx += [-0.2*ca.DM.ones(3)]
            ubx += [0.2*ca.DM.ones(3)]
    x = ca.vertcat(*x)
    x0 = ca.vertcat(*x0)
    lbx = ca.vertcat(*lbx)
    ubx = ca.vertcat(*ubx)

def package_results(res, param):
    res_dict = {}
    read_pos = 0
    for k in param.get_keys():
        v_size = param[k].size
        res_dict[k] = res[read_pos:read_pos + v_size]
        read_pos += v_size
    return res_dict

def optimize(obs, state, param):
    loss = loss_fn(obs, state, param)

    x, x0, lbx, ubx = get_dec_vectors(param)
    
    nlp = {'x':x, 'f': loss}

    opts = {'expand':False,
            'ipopt.print_level':3}

    solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    print('________________________________________')
    print(' ##### Optimizing offline params ######' )

    solve_time = -time.time()
    res = solver(x0=x0, lbx=lbx, ubx=ubx)
    status = solver.stats()['return_status']
    obj = res['f']
    hyper.set_results(res['x'])
    solve_time += time.time()

    print("Solve time:  %s - %f sec" % (status, solve_time))
    print("Final obj:  {}".format(obj))
    print()

    return 



    return param_opt



