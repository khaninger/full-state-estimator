import argparse
import pickle
from os.path import exists

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from observer import ekf
from robot import robot
from helper_fns import *
from param_fit import *

def init_rosparams(est_geom = False):
    p = {}
    p['urdf_path'] = rospy.get_param('urdf_description', 'urdf/src/racer_description/urdf/racer7.urdf')
    p['urdf'] = rospy.get_param('robot_description')

    p['fric_model']= {'visc':np.array(rospy.get_param('visc_fric', [0.2]*6))}
    p['h'] = rospy.get_param('obs_rate', 1./475.)

    p['proc_noise'] = {'pos':np.array(rospy.get_param('pos_noise', [1e-1]*6)),
                       'vel':np.array(rospy.get_param('vel_noise', [1e2]*6))}
    p['cov_init'] = np.array(rospy.get_param('cov_init', [1.]*12))
    if est_geom:
        p['proc_noise']['geom'] = np.array(rospy.get_param('geom_noise', [1]*3))
        p['cov_init'] = np.append(p['cov_init'],[1.]*3)
    p['meas_noise'] = {'pos':np.array(rospy.get_param('meas_noise', [5e-2]*6))}
    

    p['contact_1'] = {'pos': ca.DM(rospy.get_param('contact_1_pos', [0]*3)),
                      'stiff': ca.DM(rospy.get_param('contact_1_stiff', [0]*3)),
                      'rest': ca.DM(rospy.get_param('contact_1_rest', [0.8, -0.7, 0.5]))}        
    return p

class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, joint_topic = 'joint_state',
                 force_topic = 'robot_state', est_geom = False):
        self.q = None        # joint position
        self.tau_err = None  # torque error
        self.F = None        # EE force
        self.x = None        # observer state

        self.joint_sub = rospy.Subscriber(joint_topic, JointState,
                                           self.joint_callback, queue_size=1)
        self.force_sub = rospy.Subscriber(force_topic, JointState,
                                          self.force_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('observer_jt',
                                         JointState, queue_size=1)
        self.ee_pub = rospy.Publisher('observer_ee',
                                      JointState, queue_size=1)

        params = init_rosparams(est_geom)
        
        self.observer = ekf(params,
                            np.array([-0.23, 0.71, -1.33, 0.03, 1.10, 17.03]),
                            est_geom)


    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        try:
            self.q = np.array(msg.position)#/360*2*np.pi
            self.tau_err = np.array(msg.effort)
        except:
            print("Error loading ROS message in joint_callback")
        if hasattr(self, 'observer'):
            self.observer_update()
            self.publish_state()
        
    def force_callback(self, msg):
        try:
            self.F = msg.effort[:6]
        except:
            print("Error loading ROS message in force_callback")

    def observer_update(self):
        try:
            self.x = self.observer.step(q = self.q,
                                        tau_err = self.tau_err,
                                        F = self.F)
        except Exception as e:
            print("Error in observer step")
            print(e)
            rospy.signal_shutdown("error in observer")

    def publish_state(self):
        ddq = self.x.get('ddq', np.zeros(self.observer.dyn_sys.nq))
        msg = build_jt_msg(self.x['q'], self.x['dq'], self.x.get('p')) 
        if not rospy.is_shutdown():
            self.joint_pub.publish(msg)
        x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        if not rospy.is_shutdown():
            self.ee_pub.publish(msg_ee)     
    
    def shutdown(self):
        print("Shutting down observer")

def start_node(est_geom = False):
    rospy.init_node('observer')
    node = ros_observer(est_geom = est_geom)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()

def generate_traj(bag):
    print('Generating trajectory from {}'.format(bag))
    p = init_rosparams()
    msgs = bag_loader(bag, map_joint_state, topic_name = 'joint_state')
    observer = ekf(p, msgs['pos'][:,0])

    num_msgs = len(msgs['pos'].T)
    
    states = np.zeros((observer.dyn_sys.nx, num_msgs))
    contact_pts = np.zeros((3, num_msgs))
    x_ees = []
    inputs = msgs['torque']
    
    for i in range(num_msgs):
        res = observer.step(q = msgs['pos'][:,i], tau_err = msgs['torque'][:,i])
        states[:,i] = res['xi'].flatten()
        contact_pts[:,i] = res['cont_pt'].flatten()
        x_ees += [res['x_ee']]

    fname = bag[:-4]+'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump((states, inputs, contact_pts, x_ees), f)
    print('Finished saving state trajectory of length {}'.format(len(states.T)))        
    
def param_fit(bag):
    fname = bag[:-4]+'.pkl'
    if not exists(fname):
        generate_trajectory(bag)
    print("Loading trjaectory for fitting params")
    with open(fname, 'rb') as f:
        states, inputs, _, _ = pickle.load(f)
        
    p_to_opt = {}
    p_to_opt['pos'] = ca.SX.sym('pos',3)
    p_to_opt['stiff'] = ca.SX.sym('stiff',3)
    #p_to_opt['rest'] = ca.SX.sym('rest',3)
    
    p = init_rosparams()
    rob = robot(p, p_to_opt)
    optimized_par = optimize(states.T, inputs.T, p_to_opt, rob.disc_dyn)
    for k,v in optimized_par.items():
        rospy.set_param('contact_1_'+k, v.full().tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="", help="Optimize params on this bag")
    parser.add_argument("--est_geom", default=False, action='store_true',
                        help="Estimate the contact geometry online")
    parser.add_argument("--opt_param", default=False, action='store_true',
                        help="Optimize the parameters")
    parser.add_argument("--new_traj", default=False, action='store_true',
                        help="Re-estimate the state trajectory")

    args = parser.parse_args()

    if args.new_traj:
        if args.bag == "": rospy.signal_shutdown("Need bag to gen traj from")
        generate_traj(args.bag)
        
    if args.opt_param != "":
        param_fit(args.bag)
        
    start_node(est_geom = args.est_geom)
