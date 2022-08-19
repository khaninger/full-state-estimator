import pinocchio as pin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

#import rospy

class robot(self):
    def __init__(self, urdf):
        dyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('base_link'))
        self.fwd_dyn  = ca.function.deserialize(kindyn.aba())
        self.inv_dyn  = ca.Function.deserialize(kindyn.rnea())
        self.mass_mat = ca.Function.deserialize(kindyn.ccrba())

        self.q = ca.SX.sym('q', kindyn.nq())
        self.x = self.fwd_dyn(self.q)
        
        
