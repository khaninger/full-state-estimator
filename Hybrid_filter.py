import pdb
import sys
import numpy as np


import casadi as ca
from robot import Robot
import copy
from observer import build_step_fn


class HybridParticleFilter:
    def __init__(self,  par, robot):
        """
        par: dictionary of particle filter parameters
        robot: dictionary of robot instances according to different  dynamic models

        """





        self.particles = []
        self.num_particles = par['num_particles']
        self.trans_matrix = par['transition_matrix']
        self.belief_free = par['belief_init'][0]
        self.belief_contact = par['belief_init'][1]
        self.belief_init = par['belief_init']
        self.x = {'mu': robot.xi_init, 'cov': robot.cov_init}
        self.proc_noise = robot[0].proc_noise
        self.meas_noise = robot[0].meas_noise
        self.y_hat = np.zeros((self.num_particles, 6, 1))
        self.S_t = np.zeros((self.num_particles, 6, 6))
        self.step_fn = {}
        self.modes_lst = list(robot.keys())  # getting the list of keys of robot dict, for mode sampling
        for k, v in robot.items():
            self.step_fn[k] = build_step_fn(v)


        for i in range(self.num_particles):
            self.particles.append(Particle(self.belief_init, self.x['mu'], self.x['cov'], 1/self.num_particles, sampled_mode0='free-space'))

    def propagate(self, q, tau, F=None):

        for i, particle in enumerate(self.particles):
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(self.modes_lst, p=particle.mode)
            particle.mu, particle.Sigma, self.S_t[i], self.y_hat[i] = self.step_fn[particle.sampled_mode](tau, particle.mu_prev, particle.Sigma_prev, q)
            particle.mu_prev = particle.mu
            particle.Sigma_prev = particle.Sigma





    def calc_weights(self, q):
        summation = 0
        for i,particle in enumerate(self.particles):

            particle.weight = multivariate_normal(mean=self.y_hat[i], cov=self.S_t[i]).pdf(q)
            #print(particle.weight)
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
        pos = np.zeros((self.num_particles, self.dyn_free.nq))

        weights = np.zeros(self.num_particles)



        vel = np.zeros((self.num_particles, self.dyn_free.nq))

        
        for i, particle in enumerate(self.particles):
            #print(pos[i, :])
            #print(particle.mu)
            pos[i,:] = particle.mu['q'].ravel()
            #print(pos[i,:])
            weights[i] = particle.weight
            vel[i,:] = particle.mu['dq'].ravel()
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
        self.mu = self.mu_prev = mu_0
        self.mode = self.mode_prev = mode_0
        self.Sigma = self.Sigma_prev = P0
        self.weight = weight0
        self.sampled_mode = sampled_mode0







