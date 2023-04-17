import argparse
import pickle
from os.path import exists

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import casadi as ca
from observer import ekf
from Hybrid_filter import HybridParticleFilter
from robot import Robot, RobotDict
from helper_fns import *
from param_fit import *
import time

def init_rosparams():
    p = {}
    p['urdf_path'] = rospy.get_param('urdf_description', 'urdf/src/universal_robot/ur_description/urdf/ur16e.urdf')
    p['urdf'] = rospy.get_param('robot_description')
    p['joint_names'] = [ 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    p['fric_model']= {'visc':np.array(rospy.get_param('visc_fric', [0.4]*6))}
    p['h'] = rospy.get_param('obs_rate', 1./500.)

    p['proc_noise'] = {'q':  np.array(rospy.get_param('pos_noise', [1e-2]*6)),
                       'dq':  np.array(rospy.get_param('vel_noise', [1e5]*6)),
                       'pos': np.array(rospy.get_param('geom_noise', [1e1]*3)),
                       'stiff':np.array(rospy.get_param('stiff_noise', [8e6]*3))}
    p['cov_init'] = {'q': [1e-2]*6,
                     'dq': [1e5]*6,
                     'pos':[1.5e6]*3,
                     'stiff':[6e6]*3}
    p['meas_noise'] = {'pos':np.array(rospy.get_param('meas_noise', [1e-1]*6))}
    p['contact_models'] = ['contact_1']
    p['contact_1_pos']   = ca.DM(rospy.get_param('contact_1_pos', [0.1]*3))
    p['contact_1_stiff'] = ca.DM(rospy.get_param('contact_1_stiff', [100]*3))
    p['contact_1_rest']  = ca.DM(rospy.get_param('contact_1_rest', [-0.4, 0.3, 0.12]))
    p['S_t0'] = {'pos':np.array(rospy.get_param('meas_noise', [1e-1]*6))}
    p['mom_obs_K'] = [20]*6
    p['q0'] = np.array([2.29, -1.02, -0.9, -2.87, 1.55, 0.56])
    p['num_particles'] = 20
    p['belief_init'] = np.array([0.8, 0.2])
    p['transition_matrix'] = np.array([[0.8, 0.2], [0.2, 0.8]])
    p['sampled_mode'] = np.random.choice(['free-space', 'contact'], p=p['belief_init'])
    p['weight0'] = 1/p['num_particles']

    return p

class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, joint_topic = 'joint_states',
                 force_topic = 'wrench', est_pars = {}):
        
        self.q = None        # joint position
        self.tau_err = None  # torque error
        self.tau = None      # motor torque
        self.F = None        # EE force
        self.x = None        # observer state

        self.joint_sub = rospy.Subscriber(joint_topic, JointState,
                                           self.joint_callback, queue_size=1)
        self.force_sub = rospy.Subscriber(force_topic, WrenchStamped,
                                          self.force_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('observer_jt',
                                         JointState, queue_size=1)
        self.ee_pub = rospy.Publisher('observer_ee',
                                      JointState, queue_size=1)
        self.f_ee_obs_pub = rospy.Publisher('force_ee_obs', JointState, queue_size=1)
        self.f_ee_mo_pub  = rospy.Publisher('force_ee_mo',  JointState, queue_size=1)

        #self.params = init_rosparams()
        self.params = RobotDict(["full-state-estimator/config_files/contact.yaml", "full-state-estimator/config_files/free_space.yaml"], est_pars).param_dict

        #self.robot = Robot(self.params, est_pars = est_pars)
        #self.observer = ekf(self.robot)
        self.observer = HybridParticleFilter(self.params)
        
    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        try:
            q, _, current = map_ur_joint_state(msg)
            self.q = np.array(q)

        except:
            print("Error loading ROS message in joint_callback")
        if hasattr(self, 'observer'):
            self.observer_update()
            self.publish_state()

    def force_callback(self, msg):
        try:
            self.F =  np.vstack((msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                                 msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z))
        except:
            print("Error loading ROS message in force_callback")
 
    def observer_update(self):
        self.x = self.observer.step(q = self.q,
                                    tau = self.tau,
                                    F = self.F)

    def publish_state(self):
        ddq = self.x.get('ddq', np.zeros(self.observer.dyn_sys.nq))
        msg = build_jt_msg(self.x['q'], self.x['dq'],
                           np.concatenate((self.x.get('stiff',[]), self.x.get('cont_pt', []))))
        msg_f = build_jt_msg(q = self.x['f_ee_mo'], dq = self.x['f_ee_obs'], tau = self.F) 
        if not rospy.is_shutdown():
            self.joint_pub.publish(msg)
            self.f_ee_obs_pub.publish(msg_f)
        #x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        #msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        #if not rospy.is_shutdown():
            #self.ee_pub.publish(msg_ee)
    def shutdown(self):
        print("Shutting down observer")

def start_node(est_pars):
    rospy.init_node('observer')
    node = ros_observer(est_pars)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()

def generate_traj(bag, est_pars = {}):
    print('Generating trajectory from {}'.format(bag))
    #p = init_rosparams()
    msgs = bag_loader(bag, map_joint_state, topic_name = 'joint_states')
    force_unaligned = bag_loader(bag, map_wrench, topic_name = 'wrench')
    force = get_aligned_msgs(msgs, force_unaligned)

    #robot = Robot(p, est_pars = est_pars)
    #observer = ekf(robot)
    params = RobotDict(["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars)
    observer = HybridParticleFilter(params)
    num_msgs = len(msgs['pos'].T)

    sd_initial = observer.get_statedict()
    results = {k:np.zeros((v.shape[0], num_msgs)) for k,v in sd_initial.items()}
    results['true_pos'] = msgs['pos']
    results['true_vel'] = msgs['vel']
    results['f_ee'] = force['force']
    results['input'] = msgs['torque']
    print("Results dict has elements: {}".format(results.keys()))
    

    f_ee_mo = np.zeros((3, num_msgs))

    update_freq = []

    #for i in range(2000):
        #res = observer.step(q = msgs['pos'][:,0], tau = msgs['torque'][:,0])
    
    for i in range(num_msgs):
        #if i == 1 or i == 1000 or i == 3000:
        #    print(observer.cov)

        tic = time.perf_counter()
        res = observer.step(q = msgs['pos'][:,i], tau = msgs['torque'][:,i])[0]
        toc = time.perf_counter()
        update_freq.append(1/(toc-tic))
        #print(res['mu'][:6])
        statedict = observer.get_statedict()
        for k,v in statedict.items():
            results[k][:,[i]] = v

    average_freq = (sum(update_freq)/num_msgs)/1000
    print("Average update frequency is {} kHz".format(average_freq))
    fname = bag[:-4]+'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
    print('Finished saving state trajectory of length {}'.format(num_msgs))

def param_fit(bag):
    fname = bag[:-4]+'.pkl'
    if not exists(fname):
        generate_traj(bag)
    print("Loading trjaectory for fitting params")
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    states = results['xi']
    inputs = results['input']

    p_to_opt = {}
    p_to_opt['contact_1_pos'] = ca.SX.sym('pos',3)
    p_to_opt['contact_1_stiff'] = ca.SX.sym('stiff',3)
    p_to_opt['contact_1_rest'] = ca.SX.sym('rest',3)

    p = init_rosparams()
    prediction_skip = 1
    p['h'] *= prediction_skip
    rob = Robot(p, opt_pars = p_to_opt)
    optimized_par = optimize(states.T, inputs.T, p_to_opt, rob, prediction_skip)
    for k,v in optimized_par.items():
        rospy.set_param('contact_1_'+k, v.full().tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="", help="Optimize params on this bag")
    parser.add_argument("--est_geom", default=False, action='store_true',
                        help="Estimate the contact geometry online")
    parser.add_argument("--est_stiff", default=False, action='store_true',
                        help="Estimate the contact stiffness online")
    parser.add_argument("--opt_param", default=False, action='store_true',
                        help="Optimize the parameters")
    parser.add_argument("--new_traj", default=False, action='store_true',
                        help="Re-estimate the state trajectory")

    args = parser.parse_args()

    est_pars = {}
    if args.est_stiff: est_pars['contact_1'] = ['stiff']
    if args.est_geom: est_pars['contact_1'] = ['pos']

    if args.new_traj:
        if args.bag == "": rospy.signal_shutdown("Need bag to gen traj from")
        generate_traj(args.bag, est_pars)
    elif args.opt_param:
        if args.bag == "": rospy.signal_shutdown("Need bag to optimize params from")
        param_fit(args.bag)
        generate_traj(args.bag)
    else:
        start_node(est_pars)
