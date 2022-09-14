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

def init_rosparams():
    p = {}
    p['urdf_path'] = rospy.get_param('urdf_description', 'urdf/src/racer_description/urdf/racer7.urdf')
    p['urdf'] = rospy.get_param('robot_description')

    p['fric_model']= {'visc':np.array(rospy.get_param('visc_fric', [0.2]*6))}
    p['h'] = rospy.get_param('obs_rate', 1./475.)

    p['proc_noise'] = {'pos':np.array(rospy.get_param('pos_noise', [1e-1]*6)),
                       'vel':np.array(rospy.get_param('vel_noise', [1e2]*6))}
    p['meas_noise'] = {'pos':np.array(rospy.get_param('meas_noise', [5e-2]*6))}
    p['cov_init'] = np.array(rospy.get_param('cov_init', [1.]*12))

    p['contact_1'] = {'pos': np.array(rospy.get_param('contact_1_pos', [0]*3)),
                      'stiff': np.array(rospy.get_param('contact_1_stiff', [0]*3)),
                      'rest': np.array(rospy.get_param('contact_1_rest', [0]*3))}        
    return p

class ros_observer():
    """ This handles the ros interface, loads models, etc
    """
    def __init__(self, joint_topic = 'joint_state', force_topic = 'robot_state'):
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

        params = init_rosparams()
                
        self.observer = ekf(params, np.zeros(6))#self.q)


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
        msg = build_jt_msg(self.x['q'], self.x['dq'], ddq) 
        if not rospy.is_shutdown():
            self.joint_pub.publish(msg)
        x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        msg_ee = build_jt_msg(x[0].full(), dx.full(), ddx.full())
        if not rospy.is_shutdown():
            self.ee_pub.publish(msg_ee)     
    
    def shutdown(self):
        print("Shutting down observer")

def start_node():
    rospy.init_node('observer')
    node = ros_observer()
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()

def offline_observer_run(bag):
    print('Starting offline observer from {}'.format(bag))
    p = init_rosparams()
    msgs = bag_loader(bag, map_joint_state, topic_name = 'joint_state')
    observer = ekf(p, msgs['pos'][:,0])

    num_msgs = len(msgs['pos'].T)
    
    states = np.zeros((observer.dyn_sys.nx, num_msgs))
    inputs = msgs['torque']
    #for q, tau_err in zip(msgs['pos'].T, msgs['torque'].T):
    for i in range(num_msgs):
        res = observer.step(q = msgs['pos'][:,i], tau_err = msgs['torque'][:,i])
        states[:,i] = res['xi'].flatten()
    print('Finished producing state trajectory of length {}'.format(len(states.T)))
    return states, inputs

def param_fit(states, inputs):
    p_to_opt = {'pos':ca.SX.sym('pos',3),
                'stiff':ca.SX.sym('stiff',3),
                'rest':ca.SX.sym('rest',3)}
    p = init_rosparams()
    
    rob = robot(p, p_to_opt)
    optimized_par = optimize(states.T, inputs.T, p_to_opt, rob.disc_dyn)
    return optimized_par
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_param_bag", default="", help="Optimize params on this bag")
    args = parser.parse_args()

    if args.opt_param_bag != "":
        if not exists('trajectory.pkl'):
            states, inputs = offline_observer_run(args.opt_param_bag)
            with open('trajectory.pkl', 'wb') as f:
                pickle.dump((states, inputs), f)
        with open('trajectory.pkl', 'rb') as f:
            states, inputs = pickle.load(f)
        params = param_fit(states, inputs)
        for k,v in params.items():
            rospy.set_param('contact_1_'+k, v.full().tolist())
    
    start_node()
