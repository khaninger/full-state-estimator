import numpy as np
import casadi as ca
import yaml
import rosbag
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
# Motor params for ur16, found in .ur_control for joint sizes 4,4,3,2,2,2
p = {}
#p['gearratio'] = np.array(rospy.get_param('gearratio', [101, 101, 101, 54, 54, 54 ]))
#p['torque_constant'] = np.array(rospy.get_param('torque_constant',
#                                                   [0.11968, 0.11968, 0.098322,
#                                                     0.10756, 0.10756, 0.10756 ]))
p['names'] = ['shoulder_pan_joint', 'shoulder_lift_joint',
               'elbow_joint', 'wrist_1_joint',
               'wrist_2_joint', 'wrist_3_joint']
p['names_franka'] = ['panda_joint1', 'panda_joint2',
               'panda_joint3', 'panda_joint4',
               'panda_joint5', 'panda_joint6', 'panda_joint7']

def build_jt_msg(q, dq = [], tau = []):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.position = q
    msg.velocity = dq
    msg.effort = tau
    return msg


def get_pose_msg(position = None, frame_id = 'panda_link0'):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    if position is not None:
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
    return msg

def msg_to_state(msg):
    q = np.array([msg.transform.rotation.w,
         msg.transform.rotation.x,
         msg.transform.rotation.y,
         msg.transform.rotation.z])
    r = quat_to_rotvec(q)
    p = np.array([msg.transform.translation.x,
         msg.transform.translation.y,
         msg.transform.translation.z])
    return np.hstack((p.T,np.squeeze(r)))



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
    q,v,t = map_franka_joint_state(msg)
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

def map_franka_joint_state(msg):
    q = []
    v = []
    tau = []
    for jt_name in p['names_franka']:
        ind = msg.name.index(jt_name)
        q.append(msg.position[ind])
        v.append(msg.velocity[ind])
        tau.append(msg.effort[ind])
    return q, v, tau

def yaml_load(path, fi, default_path = 'config_files/'):
    try:
        yaml_file = open(path+fi, 'r')
        print("File {} loaded from {}".format(fi, path))
    except FileNotFoundError:
        print("File {} not found in {} -> loading default in {}".format(fi, path, default_path))
        yaml_file = open(default_path+fi, 'r')

    yaml_content = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)
    local_list = []
    for key, value in yaml_content.items():
        local_list.append((key, value))
    return dict(local_list)



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


def compliance_to_world(init_pose, x, only_position=False):
    # Translate x from being in init_pose frame to world frame.
    #R = rotvec_to_rotation(init_pose[3:])
    #x_w = R@x[:3]+init_pose[:3]
    #if self.mpc_params['enable_rotation']:
    #    x_w = ca.vertcat(x_w, R.T@x[3:])
    #return x_w
    # Old method!
    q0 = rotvec_to_quat(init_pose[3:])           # Initial robot orientation, quaternion
    x_w = quat_vec_mult(q0, x[:3])+init_pose[:3] # Linear position in world coords
    if not only_position and x.size()[0] == 6:
        x_w = ca.vertcat(x_w, quat_to_rotvec(quat_quat_mult(xyz_to_quat(x[3:]), q0)))
    return x_w


def rotvec_to_quat(r):
    norm_r = ca.norm_2(r)
    th_2 = norm_r/2.0
    return ca.vertcat(ca.cos(th_2),
                      ca.sin(th_2)*r[0]/norm_r,
                      ca.sin(th_2)*r[1]/norm_r,
                      ca.sin(th_2)*r[2]/norm_r)

def xyz_to_quat(xyz): # Possible to optimize?
    ty = ca.SX if type(xyz) is ca.SX else ca.DM
    q0 = ca.horzcat(ca.cos(xyz[0]/2), ca.sin(xyz[0]/2), ty(0.0), ty(0.0))
    q1 = ca.horzcat(ca.cos(xyz[1]/2), ty(0.0), ca.sin(xyz[1]/2), ty(0.0))
    q2 = ca.horzcat(ca.cos(xyz[2]/2), ty(0.0), ty(0.0), ca.sin(xyz[2]/2))
    # extrinsic = intrinsic with reversed order
    return quat_quat_mult(q0,quat_quat_mult(q1,q2))

def quat_quat_mult(q,p):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(4)
    else:
        ret = ca.DM.zeros(4)
    ret[0] = q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3]
    ret[1] = q[0]*p[1]+q[1]*p[0]-q[2]*p[3]+q[3]*p[2]
    ret[2] = q[0]*p[2]+q[1]*p[3]+q[2]*p[0]-q[3]*p[1]
    ret[3] = q[0]*p[3]-q[1]*p[2]+q[2]*p[1]+q[3]*p[0]
    return ret



def quat_vec_mult(q,v):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(3)
    elif type(q) is ca.MX:
        ret = ca.MX.zeros(3)
    else:
        ret = ca.DM.zeros(3)
    ret[0] =    v[0]*(q[0]**2+q[1]**2-q[2]**2-q[3]**2)\
             +2*v[1]*(q[1]*q[2]-q[0]*q[3])\
             +2*v[2]*(q[0]*q[2]+q[1]*q[3])
    ret[1] =  2*v[0]*(q[0]*q[3]+q[1]*q[2])\
             +  v[1]*(q[0]**2-q[1]**2+q[2]**2-q[3]**2)\
             +2*v[2]*(q[2]*q[3]-q[0]*q[1])
    ret[2] =  2*v[0]*(q[1]*q[3]-q[0]*q[2])\
             +2*v[1]*(q[0]*q[1]+q[2]*q[3])\
             +  v[2]*(q[0]**2-q[1]**2-q[2]**2+q[3]**2)
    return ret



def quat_to_rotvec(q):
    q *= ca.sign(q[0])  # multiplying all quat elements by negative 1 keeps same rotation, but only q0 > 0 works here
    th_2 = ca.acos(q[0])
    th = th_2*2.0
    rotvec = ca.vertcat(q[1]/ca.sin(th_2)*th, q[2]/ca.sin(th_2)*th, q[3]/ca.sin(th_2)*th)
    return rotvec
