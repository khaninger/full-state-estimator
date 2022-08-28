import numpy as np
import rospy
from sensor_msgs.msg import JointState

from observer import ekf

def build_jt_msg(q, dq, tau = None):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.position = q
    msg.velocity = dq
    msg.effort = tau
    return msg

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

        self.init_rosparams()
        
        self.observer = ekf(self.urdf, self.h, self.q, self.cov_init,
                            self.proc_noise, self.meas_nosie)

    def init_rosparams(self):
        self.h = rospy.get_param('obs_rate', 1./475.)
        self.urdf = rospy.get_param('robot_description')
        self.proc_noise = rospy.get_param('proc_noise', [0.01]*12)
        self.meas_nosie = rospy.get_param('meas_noise', [0.1]*6)
        cov_init = rospy.get_param('cov_init', [1.]*12)
        self.cov_init = np.diag(cov_init)
        print("Waiting to conect to ros topics...")
        while True:
            if self.q is not None: break
        print("Connected to ros topics")

    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        try:
            self.q = np.array(msg.position)
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
        self.x = self.observer.step(q = self.q,
                                    tau_err = self.tau_err,
                                    F = self.F)

    def publish_state(self):
        ddq = self.x.get('ddq', np.zeros(self.observer.dyn_sys.nq))
        msg = build_jt_msg(self.x['q'], self.x['dq'], ddq) 
        if not rospy.is_shutdown():
            self.joint_pub.publish(msg)
        x, dx, ddx = self.observer.dyn_sys.get_tcp_motion(self.x['q'], self.x['dq'], ddq)
        msg_ee = build_jt_msg(x, dx, ddx)
        if not rospy.is_shutdown():
            self.ee_pub.publish(msg_ee)     
    
    def shutdown(self):
        print("Shutting down observer")

def start_node():
    rospy.init_node('observer')
    
    node = ros_observer()
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()


    
if __name__ == '__main__':
    start_node()
