import argparse
import pickle
from os.path import exists

import numpy as np
import rospy
import tf2_ros as tf
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
import casadi as ca
from observer import ekf
from Hybrid_filter import HybridParticleFilter
from mpc_planner import MpcPlanner
from robot import Robot, RobotDict
from helper_fns import *
from param_fit import *


class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, mpc_path, joint_topic = 'joint_states', est_pars = {}):
        # loading configuration files for MPC problem
        self.mpc_params = yaml_load(mpc_path, 'mpc_params.yaml')
        self.icem_params = yaml_load(mpc_path, 'icem_params.yaml')
        self.ipopt_options = yaml_load(mpc_path, 'ipopt_options.yaml')

        self.q_m = None        # measured joint position
        self.tau_m = None      # measured joint torque
        self.x = None          # observer state

        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        self.joint_sub = rospy.Subscriber(joint_topic, JointState,
                                          self.joint_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('belief_obs',
                                         JointState, queue_size=1)
        self.F_pub = rospy.Publisher('est_force', JointState, queue_size=1)    # publisher for estimated external forces
        #self.imp_rest_pub = rospy.Publisher('cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=1)  # impedance rest point publisher
        self.imp_rest_pub = rospy.Publisher('mpc_equilibrium_pose', PoseStamped, queue_size=1)  # impedance rest point publisher

        self.robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars).param_dict
        self.ny = self.robots['free'].ny
        self.nq = self.robots['free'].nq
        self.nx = self.robots['free'].nx

        print("Building observer")
        #self.observer = ekf(self.robots['free'])
        #self.observer = ekf(self.robots['contact'])
        self.observer = HybridParticleFilter(self.robots)
        print("Observer ready to recieve msgs")

        # set up robot state and MPC state
        self.rob_state = {}
        self.mpc_state = {}
        #self.rob_state['imp_stiff'] = self.mpc_params['imp_stiff']
        self.rob_state['des_pose'] = self.mpc_params['des_pose']  # this is the desired pose for stage cost tracking term
        self.rob_state['imp_stiff'] = self.mpc_params['imp_stiff']
        self.rob_state.update(self.observer.get_statedict()[0])
        self.par_icem = {'des_pose': self.mpc_params['des_pose'], 'imp_stiff': self.mpc_params['imp_stiff']}
        self.par_icem.update(self.observer.get_statedict()[1])
        # init MPC
        self.mpc = MpcPlanner(mpc_params=self.mpc_params,
                              icem_params=self.icem_params,
                              ipopt_options=self.ipopt_options)
        self.par_client = dynamic_reconfigure.client.Client( "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node")
        self.init_orientation = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0),
                                                                rospy.Duration(1)).transform.rotation

        # Performance profiling
        self.timelist = []

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
            #print(self.x['F_ext'])
            #print(self.x['est_force'])
            #print(np.all(np.linalg.det(self.x['cov'])>0))
            #print(self.x['est_force'])
            #print('meas_force')
            #print(self.x['y_meas'][-self.nq:])
            #print(self.x['tau_g'])
            #print(-self.x['tau_ext'] + self.x['tau_g'])
            #print(self.x['tau_ext'])
            #print(self.x['y_meas'][-self.nq:])
            #print(self.x['belief_free'], self.x['belief_contact'])
            self.publish_belief()

    def observer_update(self):
        self.x = self.observer.step(q = self.q_m,
                                    tau = self.tau_m)


    def publish_belief(self):
        #ddq = self.x.get('ddq', np.zeros(self.observer.nq))
        #msg = build_jt_msg(self.x['q'], self.x['dq'],
                           #np.concatenate((self.x.get('stiff',[]), self.x.get('cont_pt', []))))
        msg = build_jt_msg([self.x['mu'][:self.nq], self.x['mu'][-self.nq:]])
        msg_belief = build_jt_msg([self.x['belief_free'], self.x['belief_contact']])  # message for belief
        msg_est_F_i = build_jt_msg(self.x['F_ext'])   # message for estimated external contact forces


        #x, dx = self.robots['free'].get_tcp_motion(self.x['mu'][:self.nq], self.x['mu'][-self.nq:])
        #msg_ee = build_jt_msg((x[0].full(), dx.full()))
    
        if not rospy.is_shutdown():

            self.joint_pub.publish(msg_belief)
            self.F_pub.publish(msg_est_F_i)
            #self.joint_pub.publish(msg)
            #self.joint_pub.publish(msg_tau_i)
            #self.ee_pub.publish(msg_ee)
            #self.f_ee_obs_pub.publish(msg_f)
        #x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        #msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        #if not rospy.is_shutdown():
            #self.ee_pub.publish(msg_ee)

    def publish_imp_rest(self):
        action_to_execute = self.mpc_state['imp_rest'][:, 0]  # mpc solver returns the complete action sequence, need to pick first element
        des_pose_w = compliance_to_world(self.rob_state['pose'], action_to_execute, only_position=True)
        msg_imp_xd = get_pose_msg(position=des_pose_w, frame_id='panda_link0')    # get desired rest pose in world frame
        msg_imp_xd.pose.orientation = self.init_orientation
        if not rospy.is_shutdown():
            self.imp_rest_pub.publish(msg_imp_xd)

    def update_state_async(self):
        pose_msg = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0), rospy.Duration(0.05))
        self.rob_state['pose'] = msg_to_state(pose_msg)

        imp_pars = self.par_client.get_configuration()   # set impedance stiffness values
        self.rob_state['imp_stiff'] = np.array((imp_pars['translational_stiffness_x'],
                                                imp_pars['translational_stiffness_y'],
                                                imp_pars['translational_stiffness_z']))
        self.icem_params['imp_stiff'] = np.array((imp_pars['translational_stiffness_x'],
                                                  imp_pars['translational_stiffness_y'],
                                                  imp_pars['translational_stiffness_z']))

    def control(self):
        if any(el is None for el in self.rob_state.values()) or rospy.is_shutdown(): return

        # MPC calc
        # Build parameters dictionary for the MPC problem
        params_mpc = self.rob_state
        params_mpc.update(self.observer.get_statedict()[0])
        params_icem = self.par_icem
        params_icem.update(self.observer.get_statedict()[1])


        start = time.time()
        self.mpc_state = self.mpc.solve(params_mpc, params_icem)
        self.timelist.append(time.time() - start)
        self.publish_imp_rest()  # publish impedance optimized rest pose --> to be sent to franka impedance interface



    def shutdown(self):
        print("Shutting down observer")

def start_node(mpc_path, est_pars):
    rospy.init_node('observer')
    node = ros_observer(mpc_path=mpc_path, est_pars=est_pars)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.sleep(1e-1)  # Sleep so ROS can init
    #rospy.spin()
    while not rospy.is_shutdown():
        node.update_state_async()
        node.control()
        time.sleep(1e-8)  # Sleep so ROS subscribers can update

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
        #print(msgs['torque'][:,i])
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
    print(min(tau_ms[1,:]))
    print(max(tau_ms[1,:]))

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
    mpc_path = "/home/ipk410/converging/full_state_estimator/config_files"
    if args.est_stiff: est_pars['contact_1'] = ['stiff']
    if args.est_geom: est_pars['contact_1'] = ['pos']

    if args.new_traj:
        if args.bag == "": rospy.signal_shutdown("Need bag to gen traj from")
        generate_traj(args.bag, est_pars)
    elif args.opt_param:
        if args.bag == "": rospy.signal_shutdown("Need bag to optimize params from")
        param_fit(args.bag)
    else:
        start_node(mpc_path=mpc_path, est_pars=est_pars)
