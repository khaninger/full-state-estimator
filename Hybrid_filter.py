import pdb
import sys
import numpy as np
import copy
from new_params import *
from scipy.stats import multivariate_normal
from observer import ekf
from robot import robot

class HybridParticleFilter:
    def __init__(self, mu_0, mode_0, )














        self.ekf = ekf(self.params,
                            np.array([2.29, -1.02, -0.9, -2.87, 1.55, 0.56]),
                            est_geom, est_stiff)
        for i in range(self.num_particles):
            self.particles.append(Particle(mode_0, mu_0, P0, 1/N, y_hat0, S_t0, np.random.choice(['free-space', 'contact'], p=mode_0)))
    def step(self):
        for particle in self.particles:
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(['free-space', 'contact'], p=particle.mode)
            if particle.sampled_mode == 'free-space':
                particle.mu = ekf.step(          dyn_sys=robot(par))[0]
                particle.Sigma = ekf.step(          dyn_sys=robot(par))[2]
                particle.S_t = ekf.step(          dyn_sys=robot(par))[4]
                particle.y_hat = ekf.step(          dyn_sys=robot(par))[5]
            elif particle.sampled_mode == 'contact':
                particle.mu = ekf.step(dyn_sys=robot(par, est_par = est_par))[0]
                particle.Sigma = ekf.step(dyn_sys=robot(par, est_par = est_par))[2]
                particle.S_t = ekf.step(dyn_sys=robot(par, est_par = est_par))[4]
                particle.y_hat = ekf.step(dyn_sys=robot(par, est_par = est_par))[5]

    def calc_weights(self, obs):
        summation = 0
        for particle in self.particles:
            particle.weight = multivariate_normal(mean=particle.y_hat, cov=particle.S_t).pdf(obs)
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
            pos[i] = particle.mu[0]
            weights[i] = particle.weight
            vel[i] = particle.mu[1]
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

    def RunFilter(self, z):
        self.PropagateState()
        self.CalcWeights(z)
        self.MultinomialResample()
        self.EstimateState()
        return self.x_hat, self.x_dot_hat, self.belief_free, self.belief_contact
class Particle:
    def __init__(self):



