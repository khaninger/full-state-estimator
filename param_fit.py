import numpy as np
import casadi as ca
import time

def loss_fn(states, inputs, param, disc_dyn):
    ''' States is a trajectory as list.
        Param are the parameters to optimize
        disc_dyn is a function for next state as fn of state, input, param '''
    pred_state =  states[0]  # pred state will be the predicted next state based on dyn
    loss = 0
    p = param
    for state, inp in zip(states, inputs):
        loss += ca.norm_2(state-pred_state)
        p['xi'] = state
        p['tau_err'] = inp

        pred_state = disc_dyn.call(p)['xi_next']

    del p['xi']
    del p['tau_err']
    
    return loss

def get_dec_vectors(param):
    x = []
    x0 = []
    lbx = []
    ubx = []
    for k in param.keys():
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
    print(x)
    x = ca.vertcat(*x)
    x0 = ca.vertcat(*x0)
    lbx = ca.vertcat(*lbx)
    ubx = ca.vertcat(*ubx)
    return x, x0, lbx, ubx

def package_results(res, param):
    res_dict = {}
    read_pos = 0
    for k in param.get_keys():
        v_size = param[k].size
        res_dict[k] = res[read_pos:read_pos + v_size]
        read_pos += v_size
    return res_dict

def optimize(states, inputs, param, disc_dyn):
    loss = loss_fn(states, inputs, param, disc_dyn)

    x, x0, lbx, ubx = get_dec_vectors(param)
    print(x)
    print(x0)
    
    nlp = {'x':x, 'f': loss}

    opts = {'expand':False,
            'ipopt.print_level':3}

    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    print('________________________________________')
    print(' ##### Optimizing offline params ######' )

    solve_time = -time.time()
    res = solver(x0=x0, lbx=lbx, ubx=ubx)
    status = solver.stats()['return_status']
    obj = res['f']
    solve_time += time.time()
    res_dict = package_results(res['x'], param)
    
    print("Solve time:  %s - %f sec" % (status, solve_time))
    print("Final obj:  {}".format(obj))
    print("Fit param:  {}".format(res_dict))

    return res_dict



