import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import JointState

# Motor params for ur16, found in .ur_control for joint sizes 4,4,3,2,2,2
p = {}
p['gearratio'] = np.array(rospy.get_param('gearratio', [101, 101, 101, 54, 54, 54 ])) 
p['torque_constant'] = np.array(rospy.get_param('torque_constant',
                                                    [0.11968, 0.11968, 0.098322,
                                                     0.10756, 0.10756, 0.10756 ]))
p['names'] =  ['shoulder_pan_joint', 'shoulder_lift_joint',
               'elbow_joint', 'wrist_1_joint',
               'wrist_2_joint', 'wrist_3_joint']

def build_jt_msg(q, dq = [], tau = []):
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

def map_wrench(msg, prev_msgs):
    if len(prev_msgs) == 0:
        prev_msgs['force'] = []
    prev_msgs['force'].append(np.hstack((msg.wrench.force.x,
                                msg.wrench.force.y,
                                msg.wrench.force.z)))
    return prev_msgs

def map_joint_state(msg, prev_msgs):
    if len(prev_msgs) == 0:
        for el in ('pos', 'vel', 'torque'):
            prev_msgs[el] = []
    q,v,t = map_ur_joint_state(msg)
    prev_msgs['pos'].append(q)
    prev_msgs['vel'].append(v)
    prev_msgs['torque'].append(t)
    return prev_msgs

def map_ur_joint_state(msg):
    q = []
    v = []
    current = []
    for jt_name in p['names']:
        ind = msg.name.index(jt_name)
        q.append(msg.position[ind])
        v.append(msg.velocity[ind])
        current.append(msg.effort[ind])
    current = np.array(current)
    motor_torque = current*p['torque_constant']
    tau = motor_torque*p['gearratio']
    return q, v, tau

def get_aligned_msgs(msgs1, msgs2):
    ''' 
    Select entries from msgs2 which occured most recently before msgs1
    '''
    aligned_msgs2 = {key:[] for key in msgs2.keys()}
    t2 = np.array(msgs2['t'])
    for t1 in msgs1['t']:
        last_before_t1 = np.where(t2<=t1)[0][-1] # find last time in t which is 
        for key in msgs2.keys():
            if key == 't': continue
            aligned_msgs2[key].append(msgs2[key][:,last_before_t1])

    for key in msgs2.keys():
        aligned_msgs2[key] = np.array(aligned_msgs2[key]).T
    
    return aligned_msgs2
