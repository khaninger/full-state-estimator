
class dyn_sys():
    """ This is an abstract class for dynamic systems """
    def __init__(self, states = {}, params = {}, input = {}, obs = {}, ct_dyn = None):
        self.states = states
        self.params = params

        self.input = input
        self.obs = obs

        self.ct_dyn = ct_dyn

    def compose(self, sys2, bindings):
        """ The composition of two dynamic systems as enforced by the equality binding pairs """
        return NotImplemented
        
    def discretize(self):
        return NotImplemented

    def get_A(self, x0, discretization = 'explicit'):
        return NotImplemented
    
    def get_C(self, x0):
        C = []
        for ob in obs:
            C += ca.jacobian(ob, self.states)
        return ca.vertstack(*C)
