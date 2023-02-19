import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import JointState

def build_jt_msg(q, dq, tau = None):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.position = q
    msg.velocity = dq
    msg.effort = tau
    return msg

def bag_loader(path, map_and_append_msg, topic_name = 'joint_state'):
    bag = rosbag.Bag(path)
    num_obs = bag.get_message_count(topic_name)
    if num_obs == 0:
        topic_name = '/'+topic_name
        num_obs = bag.get_message_count(topic_name)
    print('Loading ros bag {}  with {} msgs on topic {}'.format(path, num_obs, topic_name))

    msgs = {}
    t = []
    for _, msg, t_ros in bag.read_messages(topics=[topic_name]):
        t.append(t_ros.to_sec())
        map_and_append_msg(msg, msgs)
    t = [tt-t[0] for tt in t]

    msgs_in_order = {}
    for key in msgs.keys():
        t_in_order, el_in_order = zip(*sorted(zip(t,msgs[key])))
        msgs_in_order[key] = np.array(el_in_order).T
    msgs_in_order['t'] = np.array(t_in_order)
    
    return msgs_in_order

def map_joint_state(msg, prev_msgs):
    if len(prev_msgs) == 0:
        for el in ('pos', 'vel', 'torque'):
            prev_msgs[el] = []
    prev_msgs['pos'].append(msg.position)
    prev_msgs['vel'].append(msg.velocity)
    prev_msgs['torque'].append(msg.effort)
    return prev_msgs
