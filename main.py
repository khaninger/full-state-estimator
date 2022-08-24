import numpy as np
import rospy
from sensor_msgs.msg import JointState

from robot import robot
from observer import ekf

def build_jt_msg(q, dq, tau):
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
        self.state_init= False
        
        self.joint_sub = rospy.Subscriber(joint_topic, JointState,
                                           self.joint_callback, queue_size=1)
        self.force_sub = rospy.Subscriber(force_topic, JointState,
                                          self.force_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('obs_joint_state',
                                         JointState, queue_size=1)
   
        urdf = rospy.get_param('robot_description')
        rob = robot(urdf)

        cov_init = rospy.get_param('cov_init', [1.]*rob.nx)
        cov_init = np.diag(cov_init)
        
        self.observer = ekf(rob, self.q, cov_init)

    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        try:
            self.q = msg.position
            self.tau_err = msg.effort
            self.state_init = True
        except:
            print("Error loading ROS message in joint_callback")
        self.observer_update()

    def force_callback(self, msg):
        try:
            self.F = msg.effort[:6]
        except:
            print("Error loading ROS message in force_callback")
            
    def observer_update(self):
        if self.q and self.tau_err and self.F:
            q, dq, tau_ext = self.observer.step(q = self.q,
                                                tau_err = self.tau_err,
                                                F = self.F)
            msg = build_jt_msg(q, dq, tau_ext)
            if not rospy.is_shutdown():
                self.joint_pub.publish(msg)

    def shutdown(self):
        print("Shutting down observer")

def start_node():
    rospy.init_node('observer')
    
    node = ros_observer()
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.spin()


    
if __name__ == '__main__':
    start_node()
