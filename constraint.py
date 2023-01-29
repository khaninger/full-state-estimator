import casadi as ca
import numpy as np
from decision_vars import DecisionVar, DecisionVarSet

class Constraint():
    def __init__(self):
        self.params = None
    
    def fit(self, data):
        loss = 0
        for data_pt in data:
            loss += self.violation(data_pt)
        x, lbx, ubx = self.params.get_dec_vectors()
        x0 = self.params.get_x0()
        args = dict(x0=x0, lbx=lbx, ubx=ubx, p=None)
        prob = dict(f=loss, x=x)
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        print(self.params)

    def violation(self, x):
        # constraint violation for a single pose x
        raise NotImplementedError
    
    def get_jac(self, x):
        # jacobian evaluated at point x
        raise NotImplementedError       

    def get_similarity(self, x, f):
        raise NotImplementedError

class PointConstraint(Constraint):
    def __init__(self):
        Constraint.__init__(self)
        self.pt = ca.SX.sym('pt', 3)           # contact point, in the coordinate sys of x
        self.rest_pt = ca.SX.sym('rest_pt', 3) # rest point, where contact point should stay
        params_init = {'pt': np.zeros(3),
                       'rest_pt': np.zeros(3),}

        self.params = DecisionVarSet(x0 = params_init)
        print("Initializing a PointConstraint with following params:")
        print(self.params)

    def violation(self, x):
        x_pt = x @ ca.vertcat(self.params['pt'], ca.SX(1))
        return ca.norm_2(x_pt[:3]-self.params['rest_pt'])
        

class ConstraintSet():
    def __init__(self, dataset):
        clusters = self.cluster(dataset)
        self.constraints = []
        for cluster in clusters:
            c_type = self.id_constraint_pos(cluster)
            c_fit = self.fit_constraint(cluster, c_type)
            self.constraints.add(c_fit)
        
    def cluster(self, dataset):
        clusters = [] # list of partitioned data
        return clusters

    def fit_constraint(self, data, c_type):
        pass
    
    def id_constraint_pos(self, data):
        pass
    
    def id_constraint_force(self, x, f):
        pass
