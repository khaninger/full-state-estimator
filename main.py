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


class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, joint_topic = 'joint_states',
                       force_topic = 'wrench',
                       est_pars = {}):
        
        self.q_m = None        # measured joint position
        self.tau_m = None      # measured joint torque
        self.x = None          # observer state

        self.joint_sub = rospy.Subscriber(joint_topic, JointState,
                                          self.joint_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('joint_states_obs',
                                         JointState, queue_size=1)
        self.ee_pub = rospy.Publisher('tcp_obs',
                                      JointState, queue_size=1)

        self.robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars).param_dict
        self.ny = self.robots['free-space'].ny
        self.nq = self.robots['free-space'].nq
        self.nx = self.robots['free-space'].nx

        print("Building observer")
        #self.observer = ekf(self.robots['free-space'])
        #self.observer = ekf(self.robots['contact'])
        self.observer = HybridParticleFilter(self.robots)
        print("Observer ready to recieve msgs")
        
    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        try:
            q_m, _, tau_m = map_franka_joint_state(msg)
            self.q_m = np.array(q_m)
            self.tau_m = np.array(tau_m)
        except:
            print("Error loading ROS message in joint_callback")

        if hasattr(self, 'observer'):
            self.observer_update()
            #print(self.x['mu'][:self.nq])
            #print(self.x['mu'][-self.nq:])
            #print('est_force')
            print(self.x['est_force'])
            #print('meas_force')
            #print(self.x['y_meas'][-self.nq:])
            #print(self.x['tau_g'])
            #print(-self.x['tau_ext'] + self.x['tau_g'])
            #print(self.x['tau_ext'])
            #print(self.x['y_meas'][-self.nq:])
            #print(self.x['belief_free'], self.x['belief_contact'])
            self.publish_state()

    def observer_update(self):
        self.x = self.observer.step(q = self.q_m,
                                    tau = self.tau_m)


    def publish_state(self):
        #ddq = self.x.get('ddq', np.zeros(self.observer.nq))
        #msg = build_jt_msg(self.x['q'], self.x['dq'],
                           #np.concatenate((self.x.get('stiff',[]), self.x.get('cont_pt', []))))
        msg = build_jt_msg([self.x['mu'][:self.nq], self.x['mu'][-self.nq:]])
        msg_belief = build_jt_msg([self.x['belief_free'], self.x['belief_contact']])
        #msg_tau_i = build_jt_msg(self.x['y_meas'][-self.nq:])
        #print(msg_belief)

        x, dx = self.robots['free-space'].get_tcp_motion(self.x['mu'][:self.nq], self.x['mu'][-self.nq:])
        msg_ee = build_jt_msg((x[0].full(), dx.full()))
    
        if not rospy.is_shutdown():

            self.joint_pub.publish(msg_belief)
            #self.joint_pub.publish(msg)
            #self.joint_pub.publish(msg_tau_i)
            #self.ee_pub.publish(msg_ee)
            #self.f_ee_obs_pub.publish(msg_f)
        #x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        #msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        #if not rospy.is_shutdown():
            #self.ee_pub.publish(msg_ee)
    def shutdown(self):
        print("Shutting down observer")

def start_node(est_pars):
    rospy.init_node('observer')
    node = ros_observer(est_pars=est_pars)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()

def generate_traj(bag, est_pars = {}):
    print('Generating trajectory from {}'.format(bag))
    
    msgs = bag_loader(bag, map_joint_state, topic_name = '/joint_states')
    force_unaligned = bag_loader(bag, map_wrench, topic_name = '/franka_state_controller/F_ext')
    force = get_aligned_msgs(msgs, force_unaligned)

    #robot = Robot(p, est_pars = est_pars)

    robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars).param_dict
    observer = ekf(robots['contact'])
    #observer = ekf(robots['free-space'])
    #observer = HybridParticleFilter(robots)
    num_msgs = len(msgs['pos'].T)

    sd_initial = observer.get_statedict()
    results = {k:np.zeros((v.shape[0], num_msgs)) for k,v in sd_initial.items()}
    results['q_m'] = msgs['pos']
    results['dq_m'] = msgs['vel']
    results['f_ee'] = force['force']
    results['tau_m'] = msgs['torque']
    print("Results dict has elements: {}".format(results.keys()))
    
    update_freq = []

    for i in range(num_msgs):
        #if i == 1 or i == 1000 or i == 3000:
        #    print(observer.cov)

        tic = time.perf_counter()
        res = observer.step(q = msgs['pos'][:,i], tau = msgs['torque'][:,i])
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
    print("Loading trajectory for fitting params")
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    #states = results['xi']
    #print(np.mean(results['q_m'],axis=1))
    #print(np.mean(results['dq_m'], axis=1))
    #print(np.std(results['dq_m'], axis=1))

    states = np.vstack((results['q_m'], results['dq_m']))
    tau_ms = results['tau_m']

    p_to_opt = {}
    p_to_opt['contact_1_pos'] = ca.SX.sym('pos',3)
    p_to_opt['contact_1_stiff'] = ca.SX.sym('stiff',3)
    p_to_opt['contact_1_rest'] = ca.SX.sym('rest',3)

    robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars)
    p = robots.raw_param_dict['contact']

    prediction_skip = 1
    p['h'] *= prediction_skip
    rob = Robot(p, opt_pars = p_to_opt)
    optimized_par = optimize(states.T, tau_ms.T, p_to_opt, rob, prediction_skip)
    for k,v in optimized_par.items():
        print(f'{k}:{v}')
        #rospy.set_param('contact_1_'+k, v.full().tolist())

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
    else:
        start_node(est_pars)
