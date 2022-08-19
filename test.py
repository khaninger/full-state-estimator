import rospy
import numpy as np
from robot import robot

urdf = rospy.get_param('robot_description')
rob = robot(urdf)
f = rob.build_A(0.01)
q = np.ones(6)
print(f(0.01*q, 0*q, 0.01*q))
