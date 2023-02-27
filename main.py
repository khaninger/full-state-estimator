import argparse
import pickle
from os.path import exists

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

from observer import ekf
from robot import robot
from helper_fns import *
from param_fit import *

def init_rosparams():
    p = {}
    p['urdf_path'] = rospy.get_param('urdf_description', 'urdf/src/universal_robot/ur_description/urdf/ur16e.urdf')
    p['urdf'] = rospy.get_param('robot_description')
    p['joint_names'] = [ 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    p['fric_model']= {'visc':np.array(rospy.get_param('visc_fric', [0.4]*6))}
    p['h'] = rospy.get_param('obs_rate', 1./50.)

    p['proc_noise'] = {'pos':  np.array(rospy.get_param('pos_noise', [1e-2]*6)),
                       'vel':  np.array(rospy.get_param('vel_noise', [1e5]*6)),
                       'geom': np.array(rospy.get_param('geom_noise', [1e1]*3)),
                       'stiff':np.array(rospy.get_param('stiff_noise', [5e9]*3))}
    p['cov_init'] = {'pos': [1e-1]*6,
                     'vel': [1e5]*6,
                     'geom':[1.5]*3,
                     'stiff':[1e13]*3}
    p['meas_noise'] = {'pos':np.array(rospy.get_param('meas_noise', [1e-1]*6))}
    p['contact_1'] = {'pos':   ca.DM(rospy.get_param('contact_1_pos', [0]*3)),
                      'stiff': ca.DM(rospy.get_param('contact_1_stiff', [0]*3)),
                      'rest':  ca.DM(rospy.get_param('contact_1_rest', [-0.4, 0.3, 0.12]))}
    p['mom_obs_K'] = [20]*6
    
    return p

class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, joint_topic = 'joint_states',
                 force_topic = 'wrench', est_geom = False, est_stiff = False):
        
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

        self.params = init_rosparams()
        
        self.observer = ekf(self.params,
                            np.array([2.29, -1.02, -0.9, -2.87, 1.55, 0.56]),
                            est_geom, est_stiff)
        
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
        x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        if not rospy.is_shutdown():
            self.ee_pub.publish(msg_ee)
    def shutdown(self):
        print("Shutting down observer")

def start_node(est_geom = False, est_stiff = False):
    rospy.init_node('observer')
    node = ros_observer(est_geom = est_geom, est_stiff = est_stiff)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()

def generate_traj(bag, est_geom = False, est_stiff = False):
    print('Generating trajectory from {}'.format(bag))
    p = init_rosparams()
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
    true_pos = np.zeros((6, num_msgs))
    true_vel = np.zeros((6, num_msgs))
    x_ees = []
    inputs = msgs['torque']
    
    for i in range(num_msgs):
        true_pos[:,i] = msgs['pos'][:,i]
        true_vel[:,i] = msgs['vel'][:,i]
        res = observer.step(q = msgs['pos'][:,i], tau = msgs['torque'][:,i])
        states[:,i] = res['xi'].flatten()
        contact_pts[:,i] = res['cont_pt'].flatten()
        stiff[:,i] = res.get('stiff',np.zeros(3)).flatten()
        f_ee_mo[:,i] = res['f_ee_mo'].flatten()
        f_ee_obs[:,i] = res['f_ee_obs'].flatten()
        x_ees += [res['x_ee']]

    fname = bag[:-4]+'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump((states, inputs, contact_pts, x_ees, stiff, f_ee_mo, f_ee_obs, force['force'], true_pos, true_vel), f)
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
    prediction_skip = 1
    p['h'] *= prediction_skip
    rob = robot(p, p_to_opt)
    optimized_par = optimize(states.T, inputs.T, p_to_opt, rob.disc_dyn, prediction_skip)
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

    if args.new_traj:
        if args.bag == "": rospy.signal_shutdown("Need bag to gen traj from")
        generate_traj(args.bag, args.est_geom, args.est_stiff)
    elif args.opt_param:
        if args.bag == "": rospy.signal_shutdown("Need bag to optimize params from")
        param_fit(args.bag)
        generate_traj(args.bag)
    else:
        start_node(est_geom = args.est_geom, est_stiff = args.est_stiff)
