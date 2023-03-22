import pdb
import sys
import numpy as np
import copy

from scipy.stats import multivariate_normal
from observer import ekf
from robot import robot
import copy

class HybridParticleFilter(ekf):
    def __init__(self,  par, q0, est_geom = False, est_stiff = False):
        ekf.__init__(self, par, q0, est_geom, est_stiff)
        self.particles = []
        self.num_particles = par['num_particles']
        self.transition_matrix = par['transition_matrix']
        self.belief_free = par['belief_init'][0]
        self.belief_contact = par['belief_init'][1]
        self.belief_init = par['belief_init']

        self.dyn_free = robot(par)
        self.dyn_contact = robot(par, est_par=est_par)
        self.y_hat = np.zeros((self.num_particles, 6, 1))
        self.S_t = np.zeros((self.num_particles, 6, 6))
        for i in range(self.num_particles):
            self.particles.append(Particle(self.belief_init, self.x, self.cov, 1/self.num_particles, sampled_mode0='free-space'))
















    def propagate(self, q, tau, F=None):
        for i,particle in enumerate(self.particles):
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(['free-space', 'contact'], p=particle.mode)
            if particle.sampled_mode == 'free-space':
                particle.mu = ekf.step(q, tau, dyn_sys=self.dyn_free, F=None)[0]
                particle.Sigma = ekf.step(q,tau, dyn_sys=self.dyn_free, F=None)[1]
                self.S_t[i] = ekf.step(q, tau, dyn_sys=self.dyn_free, F=None)[2]
                self.y_hat[i] = ekf.step(q, tau, dyn_sys=self.dyn_free, F=None)[3]
            elif particle.sampled_mode == 'contact':
                particle.mu = ekf.step(q, tau, dyn_sys=self.dyn_contact, F=None)[0]
                particle.Sigma = ekf.step(q, tau, dyn_sys=self.dyn_contact, F=None)[1]
                self.S_t[i] = ekf.step(q, tau, dyn_sys=self.dyn_contact, F=None)[2]
                self.y_hat[i] = ekf.step(q, tau, dyn_sys=self.dyn_contact, F=None)[3]

    def calc_weights(self, q):
        summation = 0
        for i,particle in enumerate(self.particles):
            particle.weight = multivariate_normal(mean=self.y_hat[i], cov=self.S_t[i]).pdf(q)
            if particle.weight<1e-15:
                particle.weight = sys.float_info.epsilon
            summation += particle.weight
        self.weightsum = 0
        for particle in self.particles:
            particle.weight /= summation
            if particle.weight == 0:
                particle.weight = sys.float_info.epsilon
            self.weightsum += particle.weight
        self.belief_free = sum([particle.weight for particle in self.particles if particle.sampled_mode == 'free-space'])
        self.belief_contact = sum([particle.weight for particle in self.particles if particle.sampled_mode == 'contact'])
        for particle in self.particles:
            particle.mode_prev = np.array([self.belief_free, self.belief_contact])

    def estimate_state(self):
        pos = np.zeros(self.num_particles)
        weights = np.zeros(self.num_particles)
        vel = np.zeros(self.num_particles)
        x_env = np.zeros(self.num_particles)
        for i, particle in enumerate(self.particles):
            pos[i] = particle.mu['q']
            weights[i] = particle.weight
            vel[i] = particle.mu['dq']
        self.x_hat = np.average(pos, weights=weights, axis=0)
        self.x_dot_hat = np.average(vel, weights=weights, axis=0)
    def MultinomialResample(self):
        states_idx = []
        weights = []
        states = []
        for i in range(len(self.particles)):
            states_idx.append([self.particles[i].weight, i])
            weights.append(self.particles[i].weight)

        resamp_parts = []

        temp = np.random.multinomial(len(self.particles), weights)
        for i in range(len(temp)):
            for j in range(temp[i]):
                resamp_parts.append(copy.deepcopy(self.particles[i]))

        self.particles = resamp_parts

    def RunFilter(self, q, tau, F=None):
        self.PropagateState(q,  tau, F=None)
        self.CalcWeights(q)
        self.MultinomialResample()
        self.EstimateState()
        return self.x_hat, self.x_dot_hat, self.belief_free, self.belief_contact
class Particle:
    def __init__(self, mode_0, mu_0, P0, weight0, sampled_mode0):
        self.mode = self.mode_prev = mode_0
        self.mu = mu_0
        self.Sigma = P0
        self.weight = weight0
        self.sampled_mode = sampled_mode0





