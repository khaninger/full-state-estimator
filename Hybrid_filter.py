import pdb
import sys
import numpy as np
import copy
from new_params import *
from scipy.stats import multivariate_normal

class HybridParticleFilter:
    def __init__(self, mu_0, mode_0, )














        self.ekf = ekf(self.params,
                            np.array([2.29, -1.02, -0.9, -2.87, 1.55, 0.56]),
                            est_geom, est_stiff)