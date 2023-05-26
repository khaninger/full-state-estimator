import pdb
import sys
import numpy as np

from scipy.stats import multivariate_normal
import casadi as ca
from robot import Robot
import copy
from observer import build_step_fn


class HybridParticleFilter:
    def __init__(self, robot):
        """
        par: dictionary of particle filter parameters
        robot: dictionary of robot instances according to different  dynamic models

        """





        self.particles = []
        self.num_particles = 30
        self.trans_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
        #self.belief_init = np.array([0.8, 0.2])
        #self.belief_free = 0.8
        #self.belief_contact = 0.2
        #self.robot_dict = robot.param_dict
        self.N_eff = 0
        self.x = {'mu': robot['free-space'].xi_init, 'cov': robot['free-space'].cov_init,
                  'belief_free': 0.8, 'belief_contact': 0.2}
        self.belief_init = np.array([self.x['belief_free'], self.x['belief_contact']])
        #print(self.x['mu'])
        self.proc_noise = robot['free-space'].proc_noise
        self.meas_noise = robot['free-space'].meas_noise
        self.ny = robot['free-space'].ny
        self.nq = robot['free-space'].nq
        self.nx = robot['free-space'].nx

        self.y_hat = np.zeros((self.num_particles, self.ny, 1))
        self.S_t = np.zeros((self.num_particles, self.ny, self.ny))
        self.step_fn = {}
        self.modes_lst = list(robot.keys())  # getting the list of keys of robot dict, for mode sampling

        for k, v in robot.items():
            self.step_fn[k] = build_step_fn(v)
        #print(self.step_fn["free-space"])


        for i in range(self.num_particles):
            self.particles.append(Particle(self.belief_init, self.x['mu'], self.x['cov'], 1/self.num_particles, sampled_mode0='free-space'))

    def propagate(self, q, tau, F=None):


        for i, particle in enumerate(self.particles):
            #print(self.particles)
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(self.modes_lst, p=particle.mode)
            #print(tau.shape)
            #print(q.shape)
            particle.mu, particle.Sigma, self.S_t[i], self.y_hat[i], particle.weight = self.step_fn[particle.sampled_mode](tau, particle.mu_prev, particle.Sigma_prev, q)
            #print(self.particles)

            particle.mu_prev = particle.mu
            particle.Sigma_prev = particle.Sigma
            #print(particle.Sigma.shape)




    def calc_weights(self):
        summation = 0
        for i, particle in enumerate(self.particles):
            #print(self.y_hat[i].shape)
            particle.weight = float(particle.weight)

            #particle.weight = multivariate_normal(mean=self.y_hat[i].ravel(), cov=self.S_t[i]).pdf(q)
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
            self.weightsum += particle.weight**2
        self.N_eff = 1/self.weightsum  # effective number of particles, needed for resampling step
        self.x['belief_free'] = sum([particle.weight for particle in self.particles if particle.sampled_mode == 'free-space'])
        self.x['belief_contact'] = sum([particle.weight for particle in self.particles if particle.sampled_mode == 'contact'])
        #print(self.belief_contact)
        for particle in self.particles:
            particle.mode_prev = np.array([self.x['belief_free'], self.x['belief_contact']])


    def estimate_state(self):
        pos = ca.DM(self.num_particles, self.nq)

        weights = np.zeros(self.num_particles)
        cov = np.zeros([self.num_particles, 2*self.nq, 2*self.nq])




        vel = ca.DM(self.num_particles, self.nq)


        for i, particle in enumerate(self.particles):
            #print(pos[i, :])
            #print(particle.mu[:6].shape)
            pos[i,:] = particle.mu[:self.nq]
            #print(particle.weight)
            weights[i] = particle.weight
            vel[i,:] = particle.mu[-self.nq:]
            cov[i] = particle.Sigma
        #print(weights.shape)
        self.x['mu'][:self.nq] = np.average(pos, weights=weights, axis=0)
        self.x["mu"][-self.nq:] = np.average(vel, weights=weights, axis=0)
        self.x["cov"] = np.average(cov, weights=weights, axis=0)
        #print(self.x["cov"].shape)

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

    def StratifiedResampling(self):
        n = 0
        m = 0
        new_samples = []
        weights = [particle.weight for particle in self.particles]
        Q = np.cumsum(weights).tolist()
        while n < self.num_particles:
            u0 = np.random.uniform(1e-10, 1.0 / self.num_particles, 1)[0]
            u = u0 + float(n) / self.num_particles
            while Q[m] < u:
                m += 1
            new_samples.append(copy.deepcopy(self.particles[m]))
            n += 1

        self.particles = new_samples


    def get_statedict(self):
        return {}

    def step(self, q, tau, F=None):
        self.propagate(q, tau, F=None)
        self.calc_weights()
        if self.N_eff < self.num_particles/5:
            self.StratifiedResampling()
        self.estimate_state()
        return self.x

class Particle:
    def __init__(self, mode_0, mu_0, P0, weight0, sampled_mode0):
        self.mu = self.mu_prev = mu_0
        self.mode = self.mode_prev = mode_0
        self.Sigma = self.Sigma_prev = P0
        self.weight = weight0
        self.sampled_mode = sampled_mode0







