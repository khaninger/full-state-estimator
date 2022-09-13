import rospy
import numpy as np
import casadi as ca
from robot import robot
import pinocchio as pin
import pinocchio.casadi as cpin

import faulthandler
faulthandler.enable()

#model = pin.buildModelFromUrdf('urdf/src/racer_description/urdf/racer7.urdf')
model = pin.buildModelFromUrdf('urdf/src/ur_description/urdf/ur5_robot.urdf')
data = model.createData()
cmodel = cpin.Model(model)
cdata = cmodel.createData()


q = np.zeros((6,1))
Minv = pin.computeMinverse(model, data, q)
#print(Minv)

#print(cpin.computeMinverse)
cq = ca.SX(6,1)
Minv = cpin.computeMinverse(cmodel, cdata, cq)
print(Minv)
x = cpin.computeForwardKinematicsDerivatives(cmodel, cdata, cq, cq, cq)
print(x)
x = cpin.forwardKinematics(cmodel, cdata, cq)
print(x)


#fwd_kin = pin.forwardKinematics(cmodel, cdata, q, dq, ddq)
#print(fwd_kin)
#x = ca.jacobian(fwd_kin, q)


#urdf = rospy.get_param('robot_description')
#rob = robot(urdf)
#f = rob.build_A(0.01)
#q = np.ones(6)
#print(f(0.01*q, 0*q, 0.01*q))
