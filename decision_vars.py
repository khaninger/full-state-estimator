"""
Copyright (c) 2022, Kevin Haninger
Helper classes for decision variables in an optimization problem
"""

import casadi as ca
import numpy as np
from sys import version_info
from copy import deepcopy

class DecisionVar():
    """
    Individual optimization variable, initialized from an initial value x0.
    Upper/lower bounds ub/lb default to +/- np.inf unless overwritten
    """
    def __init__(self, x0, lb = -np.inf, ub = np.inf, x = None):
        x0 = np.array(x0)
        self.size  = x0.size

        self.x0 = x0
        self.shape = self.x0.shape
        self.lb = np.full(self.shape, lb)
        self.ub = np.full(self.shape, ub)
        self.x = x

    def __len__(self):
        return self.size

class DecisionVarSet():
    """
    Helper class for sets of decision variables.

    It's almost like a dictionary, but with some fun stuff added:
      - Add new variable to the set with dec_var_set[key] = decision_var(x0)
      - Keys have an upper and lower bound
      - Before optimization, [] gives symbolic var, afterwards gives numerical value
      - Set optimized results with dec_var_set.set_results(x_opt)
    """

    def __init__(self, symb_type = ca.SX.sym, x0 = {}, lb = {}, ub = {}):
        """
        Arguments:
          - var_type: the type of symbolic optimization variable which should be constructed
          - x0: optional dict of initial values, will construct decision_vars for all keys/values
          - ub: optional dict of upper bounds, if no key for a key in x0, will default to  np.inf
          - lb: optional dict of lower bounds, if no key for a key in x0, will default to -np.inf
        """
        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"
        self.__ty = symb_type     # Type of symbolic variable
        self.__vars = {}          # Individual variables
        self.__optimized = False  # Flag if set_results has been called

        for key in x0.keys():
            self[key] = DecisionVar(x0[key],
                                    lb = lb.get(key, -np.inf),
                                    ub = ub.get(key, np.inf))

    def __setitem__(self, key, value):
        """
        # Arguments:
            key: name of variable
            value: decision_var which should be set there
        """
        if value.x is None:
            value.x = self.__ty(key, *value.shape)
        self.__vars[key] = value
        self.__keys = list(self.__vars.keys())

    def __getitem__(self, key):
        """
        If no results are set, returns the symbolic variable at key
        If results are set, returns the numeric value at key
        """
        if not key in self.__vars: return None
        return self.__vars[key].x

    def __len__(self):
        return sum(len(val) for val in self.__vars.values())

    def __str__(self):
        s = "** Decision variables **\n"
        for key in self.__keys:
            s += "{}:\n: {}\n".format(key, self[key])
        return s

    def skip_opt(self):
        for key in self.__vars.keys():
            self.__vars[key].x = self.__vars[key].x0

    def filter(self, to_ignore = [], ignore_numeric = False):
        """
        Returns the decision variables or optimized values not in the to_ignore list.
        Ignore_numeric also drops numerical values from the filter output
        """
        filtered_dict = {}
        for key in self.__keys:
            if key in to_ignore: continue
            if ignore_numeric and (isinstance(self[key], np.ndarray) or not self[key].is_symbolic()): continue
            filtered_dict[key] = self[key]
        return filtered_dict

    def vectorize(self, attr):
        return ca.vertcat(*[getattr(el, attr).reshape((el.size,1)) for el in self.__vars.values()])

    def get_elem_vector(self, key):
        return ca.vertcat(self.__vars[key].x.reshape((self.__vars[key].size,1)))

    def get_dec_vectors(self):
        """
        Returns a tuple that you need to get that optimization problem going
        """
        x  = self.vectorize('x')
        lb = self.vectorize('lb')
        ub = self.vectorize('ub')
        return x, lb, ub

    def get_x0(self):
        return self.vectorize('x0')

    def get_deviation(self, key):
        """
        Returns difference between initial value and symbolic (or numeric) value
        """
        return self[key]-self.__vars[key].x0

    def set_x0(self, key, x0):
        self.__vars[key].x0 = x0

    def set_results(self, x_opt):
        """
        x_opt is the numerical optimization results, fills the dict x with reshaping as needed
        """
        x_opt = x_opt.full()
        assert len(x_opt) is len(self), "Length of optimization doesn't match initial x0"
        read_pos = 0
        for key in self.__keys:
            v_size  = self.__vars[key].size
            v_shape = self.__vars[key].shape
            if len(v_shape) == 1: v_shape = (*v_shape,1)
            self.__vars[key].x = np.squeeze(ca.reshape(x_opt[read_pos:read_pos+v_size], *v_shape))
            read_pos += v_size
        self.__optimized = True

def vectorize(dic):
    return ca.vertcat(*[ca.reshape(el, int(np.prod(el.shape)),1) for el in dic.values()])
