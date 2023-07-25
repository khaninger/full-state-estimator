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
        self.num_particles = 80
        self.trans_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
        #self.belief_init = np.array([0.8, 0.2])
        #self.belief_free = 0.8
        #self.belief_contact = 0.2
        #self.robot_dict = robot.param_dict
        self.N_eff = 0
        self.x = {'mu': robot['free'].xi_init, 'cov': robot['free'].cov_init,
                  'belief_free': 0.6, 'belief_contact': 0.4}
        self.belief_init = np.array([self.x['belief_contact'], self.x['belief_free']])   # be consistent with this ordering!!!!!!!!!!!!!!
        #print(self.x['mu'])
        self.proc_noise = robot['free'].proc_noise
        self.meas_noise = robot['free'].meas_noise
        self.ny = robot['free'].ny
        self.nq = robot['free'].nq
        self.nx = robot['free'].nx
        self.pinv_jac = robot['free'].jacpinv  # casadi function for the jacobian pseudoinverse, need to feed joint position vector as input
        self.fwd_kin = robot['free'].fwd_kin   # casadi function for calculating tcp motion
        self.y_hat = np.zeros((self.num_particles, self.ny, 1))
        self.S_t = np.zeros((self.num_particles, self.ny, self.ny))
        #self.F_i = np.zeros((self.num_particles, 3, 1))     # tensor for storing external force
        self.y_meas = np.zeros((self.num_particles, self.ny, 1))
        self.step_fn = {}
        self.modes_lst = list(robot.keys())  # getting the list of keys of robot dict, for mode sampling
        self.A = np.zeros((self.num_particles, self.nx, self.nx))
        self.C = np.zeros((self.num_particles, self.ny, self.ny))

        for k, v in robot.items():
            self.step_fn[k] = build_step_fn(v)
        #print(self.step_fn["free-space"])


        for i in range(self.num_particles):
            self.particles.append(Particle(self.belief_init, self.x['mu'], self.x['cov'], 1/self.num_particles, sampled_mode0='free-space'))

    def propagate(self, q, tau, F=None):


        for i, particle in enumerate(self.particles):
            #print(self.particles)
            #print(particle.mode_prev)
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            #print(particle.mode)
            #print(self.modes_lst)
            particle.sampled_mode = np.random.choice(self.modes_lst, p=particle.mode)
            #print(tau.shape)
            #print(q.shape)
            step_args = {'tau': tau,
                         'mu': particle.mu_prev,
                         'cov': particle.Sigma_prev,
                         'q_meas': q}
            res = self.step_fn[particle.sampled_mode].call(step_args)
            particle.mu = res['mu_next']
            particle.Sigma = res['cov_next']
            self.S_t[i] = res['S_hat']
            #self.F_i[i] = res['F_ext']
            self.y_hat[i] = res['y_hat']   # predicted measurements
            particle.weight = res['likelihood']
            self.y_meas[i] = res['y_meas']
            #print(particle.sampled_mode, res['F_ext'])
            #if np.linalg.norm(res['F_ext'])>200:
                #print("force > 200")
            #print(particle.sampled_mode, np.linalg.norm(res['F_ext'])>200)
            #print(particle.mu[:self.nq])
            #print(self.y_meas[i])
            #print(res['tau_g'])
            #print(particle.sampled_mode, particle.weight)
            #print(np.linalg.det(res['A']), particle.sampled_mode)
            #print(res['A'].shape)
            #print(np.linalg.det(self.S_t[i]))
            #print(np.linalg.det(res['Q']))
            #print(np.linalg.det(res['C']))
            #print(ca.det(res['C']))
            #print(self.particles)
            #print(np.linalg.det(res['cov_next_pre']), particle.sampled_mode)
            #print(np.all(np.linalg.eigvalsh(self.S_t[i])), particle.sampled_mode)
            #print(np.all(np.linalg.eigvalsh(self.S_t[i]) > 0))
            #print(particle.sampled_mode, particle.weight)
            #print(particle.sampled_mode, self.y_hat[i][-self.nq:])
            #print(particle.sampled_mode, particle.mu[:self.nq])
            #print(tau)
            #print(np.all(np.linalg.eigvalsh(particle.Sigma)>0), particle.sampled_mode)
            #print(np.linalg.det(particle.Sigma), particle.sampled_mode)
            #print(self.y_hat[i].shape)
            #print(self.step_fn[particle.sampled_mode])
            particle.mu_prev = particle.mu
            particle.Sigma_prev = particle.Sigma
            #print(particle.Sigma.shape)




    def calc_weights(self):
        summation = 0
        for i, particle in enumerate(self.particles):
            #print(self.y_hat[i].shape)
            #particle.weight = float(np.exp(particle.weight))
            #particle.weight = float(particle.weight)

            #particle.weight = multivariate_normal(mean=self.y_hat[i].ravel(), cov=self.S_t[i]).pdf(q)
            #print(particle.weight)
            #if particle.weight<1e-15:

                #particle.weight = sys.float_info.epsilon

            if particle.sampled_mode == 'free':
                particle.weight += np.log(particle.mode[1])
            elif particle.sampled_mode == 'contact':
                particle.weight += np.log(particle.mode[0])
            summation += np.exp(particle.weight)
        self.weightsum = 0
        for particle in self.particles:
            particle.weight -= np.log(summation)
            if particle.weight == 0:
                particle.weight = sys.float_info.epsilon
            #print(particle.weight)
            self.weightsum += np.exp(particle.weight)**2
        self.N_eff = 1/self.weightsum  # effective number of particles, needed for resampling step
        self.x['belief_free'] = sum([np.exp(particle.weight) for particle in self.particles if particle.sampled_mode == 'free'])
        self.x['belief_contact'] = sum([np.exp(particle.weight) for particle in self.particles if particle.sampled_mode == 'contact'])
        #print(self.x['belief_free'])
        for particle in self.particles:
            particle.mode_prev = np.array([self.x['belief_contact'], self.x['belief_free']]).ravel()
            #print(particle.mode_prev)


    def estimate_state(self):
        pos = ca.DM(self.num_particles, self.nq)
        est_force = ca.DM(self.num_particles, self.nq)
        weights = np.zeros(self.num_particles)
        cov = np.zeros([self.num_particles, 2*self.nq, 2*self.nq])






        vel = ca.DM(self.num_particles, self.nq)
        list_tuples = []


        for i, particle in enumerate(self.particles):
            #print(pos[i, :])
            #print(particle.mu[:6].shape)
            list_tuples.append((particle.sampled_mode, particle.mu))    # creating the list of tuples to be returned for CEM
            pos[i, :] = particle.mu[:self.nq]
            est_force[i, :] = self.y_hat[i][-self.nq:]

            #print(self.y_hat[i][-self.nq:])
            #print(particle.weight)
            weights[i] = np.exp(particle.weight)
            vel[i, :] = particle.mu[-self.nq:]
            cov[i] = particle.Sigma
        #print(weights.shape)
        self.x['est_force'] = np.average(est_force, weights=weights, axis=0)
        self.x['F_ext'] = np.array(self.pinv_jac(np.average(pos, weights=weights, axis=0)) @ np.average(est_force, weights=weights, axis=0)).ravel()
        self.x['mu'][:self.nq] = np.average(pos, weights=weights, axis=0)
        self.x["mu"][-self.nq:] = np.average(vel, weights=weights, axis=0)
        self.x["cov"] = np.average(cov, weights=weights, axis=0)
        self.x['y_meas'] = self.y_meas[0][-self.nq:]
        self.x['list_particles'] = list_tuples
        self.x['weights'] = weights

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
        weights = [np.exp(particle.weight) for particle in self.particles]
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
        self.estimate_state()
        state_dict = {}
        icem_param = {}
        state_dict['init_state'] = self.x['mu']
        icem_param['list_particles'] = self.x['list_particles']
        icem_param['weights'] = self.x['weights']
        state_dict['belief_free'] = self.x['belief_free']
        state_dict['belief_contact'] = self.x['belief_contact']
        return state_dict, icem_param

    def get_tcp_motion(self, q):
        x = self.fwd_kin(q)[0]       # get just cartesian position
        return x

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







