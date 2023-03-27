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
        self.trans_matrix = par['transition_matrix']
        self.belief_free = par['belief_init'][0]
        self.belief_contact = par['belief_init'][1]
        self.belief_init = par['belief_init']

        self.dyn_free = robot(par)
        self.dyn_contact = robot(par, est_par=self.est_par)
        self.y_est = np.zeros((self.num_particles, 6, 1))
        self.S_est = np.zeros((self.num_particles, 6, 6))
        for i in range(self.num_particles):
            self.particles.append(Particle(self.belief_init, self.x, self.cov, 1/self.num_particles, sampled_mode0=np.random.choice(['free-space', 'contact'], p=self.belief_init)))
















    def propagate(self, q, tau, F=None):
        for i,particle in enumerate(self.particles):
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(['free-space', 'contact'], p=particle.mode)
            if particle.sampled_mode == 'free-space':
                particle.mu = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_free)[0]
                particle.Sigma = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_free)[1]

                #print(particle.Sigma)
                self.S_est[i] = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_free)[2]

                self.y_est[i] = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_free)[3]
                #print(self.y_est[i].shape)
                #print(self.S_t[i].shape)
            elif particle.sampled_mode == 'contact':
                particle.mu = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_contact)[0]
                particle.Sigma = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_contact)[1]
                #print(particle.Sigma)
                self.S_est[i] = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_contact)[2]
                self.y_est[i] = ekf.step(self, q=q, tau=tau, dyn_sys=self.dyn_contact)[3]
            #print("debug1")
            #print(self.S_est[i])
            #print(np.linalg.eigvalsh(self.S_est[i]))
            #print("debug transpose")
            #print(print(self.S_est[i].T))
            #print(particle.Sigma)
            #print(particle.sampled_mode)

    def calc_weights(self, q):
        summation = 0
        for i,particle in enumerate(self.particles):
            #print(self.y_est[i].shape)
            particle.weight = multivariate_normal(mean=self.y_est[i].ravel(), cov=self.S_est[i]).pdf(q)
            if particle.weight<1e-15:
                particle.weight = sys.float_info.epsilon
            summation += particle.weight
        self.weightsum = 0
        for particle in self.particles:
            particle.weight /= summation
            if particle.weight == 0:
                particle.weight = sys.float_info.epsilon
            #print(particle.weight)
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
        self.propagate(q,  tau, F=None)
        self.calc_weights(q)
        self.MultinomialResample()
        self.estimate_state()
        return self.x_hat, self.x_dot_hat, self.belief_free, self.belief_contact
class Particle:
    def __init__(self, mode_0, mu_0, P0, weight0, sampled_mode0):
        self.mode = self.mode_prev = mode_0
        self.mu = mu_0
        self.Sigma = P0
        self.weight = weight0
        self.sampled_mode = sampled_mode0





