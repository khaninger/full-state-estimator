import numpy as np
import casadi as ca
import time

def loss_fn(states, torques, param, robot, prediction_skip = 1, num_pts = 2000):
    ''' States is a trajectory as list.
        Param are the parameters to optimize
        disc_dyn is a function for next state as fn of state, input, param '''
    pred_state =  states[0]  # pred state will be the predicted next state based on dyn
    loss = 0
    if num_pts > len(states):
        print("Traj only has {} points, less than requested pts. Still doing it".format(len(states)))
        num_pts = len(states)
    skip_size = int(len(states)/num_pts)
    
    cont_pts = []
    cont_pts_mean = ca.DM((0,0,0))
    for i in range(0, len(states)-prediction_skip, skip_size):
        param['xi'] = states[i]
        #param['tau'] = inputs[i]
        res_dyn = robot.disc_dyn.call(param)
        #loss += ca.norm_2(states[i+prediction_skip]-res['xi_next'])

        res_obs = robot.obs.call(param)
        loss += ca.norm_2(torques[i + prediction_skip] - res_obs['tau'])
        statedict = robot.get_statedict(res_dyn['xi_next'])
        loss += 0.1*ca.norm_2(statedict['contact_1_disp'])

    del param['xi']
    #del param['tau']

    for k,v in param.items():
        if k == 'stiff':
            loss += 0#1e-12*ca.sqrt(ca.norm_1(v))
        elif k == 'pos':
            loss += 0#50.*v.T@v #+ 100*v[0]*v[0]
    return loss

def validate(states, torques, param, robot, num_pts = 3500):
    pred_state =  states[0]  # pred state will be the predicted next state based on dyn
    num_pts = len(states)
    state_err = 0.0
    obs_err = 0.0
    for i in range(0, len(states)-1):
        param['xi'] = states[i]
        #param['tau'] = inputs[i]

        res_dyn = robot.disc_dyn.call(param)
        # loss += ca.norm_2(states[i+prediction_skip]-res['xi_next'])

        res_obs = robot.obs.call(param)
        state_err += ca.norm_2(states[i+1]-res_dyn['xi_next'])
        obs_err += ca.norm_2(torques[i]-res_obs['tau'])

    del param['xi']
    #del param['tau']

    print("At param values {}".format(param))
    print("State err: {}".format(state_err/len(states)))
    print("Torque error: {}".format(obs_err/len(states)))

def get_dec_vectors(param):
    x = []
    x0 = []
    lbx = []
    ubx = []
    
    for k in param.keys():
        x += [param[k]]
        if 'stiff' in k:
            x0 += [0.1*ca.DM.ones(3)]
            lbx += [ca.DM.zeros(3)]
            ubx += [1e6*ca.DM.ones(3)]
        if 'pos' in k:
            x0 += [ca.DM((0.1, 0.1, 0.1))]
            lbx += [ca.DM((-1.0, -1.0, 0.0))]
            ubx += [ca.DM((1.0, 1.0, 0.6))]
        if 'rest' in k:
            x0 += [ca.DM((-0.0, 0.0, 0.0))]
            lbx += [-2*ca.DM.ones(3)]
            ubx += [2*ca.DM.ones(3)]
    
    x = ca.vertcat(*x)
    x0 = ca.vertcat(*x0)
    lbx = ca.vertcat(*lbx)
    ubx = ca.vertcat(*ubx)
    
    return x, x0, lbx, ubx

def package_results(res, param):
    res_dict = {}
    read_pos = 0
    for k in param.keys():
        v_size = param[k].size()[0]
        res_dict[k] = res[read_pos:read_pos + v_size]
        read_pos += v_size
    return res_dict

def optimize(states, torques, param, robot, prediction_skip):
    loss = loss_fn(states, torques, param, robot, prediction_skip)

    x, x0, lbx, ubx = get_dec_vectors(param)
    
    nlp = {'x':x, 'f': loss}

    opts = {'expand':False,
            'ipopt.print_level':5}

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

    validate(states, torques, res_dict, robot)
    print("Fit param:  {}".format(res_dict))

    return res_dict



