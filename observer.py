import pinocchio as pin
import numpy as np
import casadi as ca
from casadi_kin_dyn import pycasadi_kin_dyn

class robot():
    def __init__(self, urdf):
        kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
        self.fwd_kin  = ca.Function.deserialize(kindyn.fk('base_link'))
        self.inv_dyn  = ca.Function.deserialize(kindyn.rnea())
        self.mass_mat = ca.Function.deserialize(kindyn.ccrba())
        self.fwd_dyn  = kindyn.aba

        self.q = ca.SX.sym('q', kindyn.nq())
        self.x = self.fwd_kin(self.q)

        self.env_dyn = {}

    def couple_env(self, env_dyn):
        

        
    def get_integrator(self):
        
