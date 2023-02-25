

def generate_traj(bag, est_geom = False, est_stiff = False):
    print('Generating trajectory from {}'.format(bag))
    p = init_rosparams(est_geom, est_stiff)
    msgs = bag_loader(bag, map_joint_state, topic_name = 'joint_states')
    force_unaligned = bag_loader(bag, map_wrench, topic_name = 'wrench')
    force = get_aligned_msgs(msgs, force_unaligned)
    
    observer = ekf(p, msgs['pos'][:,0], est_geom, est_stiff)

    num_msgs = len(msgs['pos'].T)
    
    states = np.zeros((observer.dyn_sys.nx, num_msgs))
    contact_pts = np.zeros((3, num_msgs))
    stiff = np.zeros((3, num_msgs))
    f_ee_mo = np.zeros((3, num_msgs))
    f_ee_obs = np.zeros((3, num_msgs))
    x_ees = []
    inputs = msgs['torque']
    
    for i in range(num_msgs):
        res = observer.step(q = msgs['pos'][:,i], tau = msgs['torque'][:,i])
        states[:,i] = res['xi'].flatten()
        contact_pts[:,i] = res['cont_pt'].flatten()
        stiff[:,i] = res.get('stiff',np.zeros(3)).flatten()
        f_ee_mo[:,i] = res['f_ee_mo'].flatten()
        f_ee_obs[:,i] = res['f_ee_obs'].flatten()
        x_ees += [res['x_ee']]

    fname = bag[:-4]+'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump((states, inputs, contact_pts, x_ees, stiff, f_ee_mo, f_ee_obs, force['force']), f)
    print('Finished saving state trajectory of length {}'.format(len(states.T)))

def param_fit(bag):
    fname = bag[:-4]+'.pkl'
    if not exists(fname):
        generate_traj(bag)
    print("Loading trjaectory for fitting params")
    with open(fname, 'rb') as f:
        states, inputs = pickle.load(f)[:2]

    p_to_opt = {}
    p_to_opt['pos'] = ca.SX.sym('pos',3)
    p_to_opt['stiff'] = ca.SX.sym('stiff',3)
    p_to_opt['rest'] = ca.SX.sym('rest',3)

    p = init_rosparams()
    rob = robot(p, p_to_opt)
    optimized_par = optimize(states.T, inputs.T, p_to_opt, rob.disc_dyn)
    for k,v in optimized_par.items():
        rospy.set_param('contact_1_'+k, v.full().tolist())
