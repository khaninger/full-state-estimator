import pdb
import sys
import numpy as np
import copy
import casadi as ca
from scipy.stats import multivariate_normal
from observer import ekf
from robot import robot
from observer import static_ekf_update


class HybridParticleFilter:
    def __init__(self,  par, q0, est_geom=False, est_stiff=False):
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
            self.particles.append(Particle(par, q0, est_geom, est_stiff))

    def step_ekf(self, q, tau, dyn_sys, x, cov, F=None):
        """ Steps the observer based on the input at time t and observation at time t
            Standard EKF update, see, e.g. pg. 51 in Thrun "Probabilistic Robotics" """
        step_args = {'tau': tau,
                     'xi': ca.vertcat(x['q'], x['dq'],
                                      x.get('p', []),
                                      x.get('stiff', []))}
        # print(self.cov)
        x_next = dyn_sys.disc_dyn.call(step_args)  # predict state and output at next time step
        #print(x_next)
        x_next = x_next["xi_next"]
        # print(x_next.shape)
        A, C = dyn_sys.get_linearized(step_args)  # get the linearized dynamics and observation matrices
        # print(C.shape)
        # print(f"F_i = {self.dyn_sys.jacpinv(self.x['q']).T@x_next['tau_err']}")
        # print(f"tau_err = {x_next['tau_err']}")
        # print(f"tau     = {tau}")
        # print(step_args['tau_err'])
        # print(q-x_next['xi_next'][:6])
        # print(A)
        # print(np.argwhere(np.isnan(self.cov)))
        cov_next = A @ cov @ (A.T) + self.proc_noise
        # print(C)

        # print(self.cov)
        S = C @ cov_next @ (C.T) + self.meas_noise
        y_hat = C @ x_next
        # print(self.y_hat.shape)
        # print(self.cov_next)
        L = cov_next @ C.T @ ca.inv(C @ cov_next @ (C.T) + self.meas_noise)  # calculate Kalman gain
        # print(self.L.shape)
        if np.any(np.isnan(L)): raise ValueError("Nans in the L matrix")

        xi_corr = x_next['xi_next'] + L @ (q - x_next['xi_next'][dyn_sys.nq])
        # print((L@(q - x_next['xi_next'][:self.dyn_sys.nq]))[-3:])
        # print(xi_corr)
        x['q'] = xi_corr[:dyn_sys.nq].full()
        x['dq'] = xi_corr[dyn_sys.nq:2 * dyn_sys.nq].full()
        if self.est_geom: x['p'] = xi_corr[2 * dyn_sys.nq:].full()
        if self.est_stiff: x['stiff'] = xi_corr[2 * dyn_sys.nq:].full().flatten()
        x['cont_pt'] = x_next['cont_pt'].full().flatten()
        x_ee = dyn_sys.fwd_kin(x['q'])
        x['x_ee'] = (x_ee[0].full(),
                          x_ee[1].full())
        x['xi'] = xi_corr.full()
        mom_obs.step(x_next['mom'], x_next['tau_err'])
        x['f_ee_mo'] = (dyn_sys.jacpinv(x['q']) @ mom_obs.r).full()
        x['f_ee_obs'] = -(dyn_sys.jacpinv(x['q']) @ x_next['tau_i']).full()

        cov = (ca.DM.eye(dyn_sys.nx) - L @ C) @ cov_next  # corrected covariance
        # print("debug2")
        # print(self.cov)
        # print(self.cov[-3:,-3:])
        # x_est = copy.deepcopy(self.x)
        # cov_est = copy.deepcopy(self.cov)
        # S = copy.deepcopy(self.S)
        # y_hat = copy.deepcopy(self.y_hat)
        return x, cov, S, y_hat














    def propagate(self, q, tau, F=None):

        for i,particle in enumerate(self.particles):
            particle.mode = np.matmul(particle.mode_prev, self.trans_matrix)
            particle.sampled_mode = np.random.choice(['free-space', 'contact'], p=particle.mode)
            #particle.mu = copy.deepcopy(particle.mu)
            #particle.Sigma = copy.deepcopy(particle.Sigma)
            #print("debug")
            #print(particle.mode)

            if particle.sampled_mode == 'free-space':
                #res = self.step_ekf(q=q, tau=tau, dyn_sys=self.dyn_contact, x=self.x, cov=self.cov)
                #particle.mu = self.step(q=q, tau=tau, dyn_sys=self.dyn_contact)[0]
                particle.mu, particle.Sigma, self.S_est[i], self.y_est[i] = particle.step(q=q, tau=tau, dyn_sys=self.dyn_free)
                #particle.mu, particle.Sigma, self.S_est[i], self.y_est[i] = static_ekf_update(cov=self.cov, x=self.x, q=q, tau=tau, dyn_sys=self.dyn_free, proc_noise=self.proc_noise, meas_noise=self.meas_noise, est_geom=self.est_geom, est_stiff=self.est_stiff)
                #particle.mu, particle.Sigma, self.S_est[i], self.y_est[i] = self.step_ekf(q=q, tau=tau, dyn_sys=self.dyn_free, x=self.x, cov=self.cov)

                #particle.Sigma = self.step(q=q, tau=tau, dyn_sys=self.dyn_free)[1]
                #self.S_est[i] = self.step(q=q, tau=tau, dyn_sys=self.dyn_free)[2]
                #self.y_est[i] = self.step(q=q, tau=tau, dyn_sys=self.dyn_free)[3]
                #print(self.y_est[i].shape)
                #print(self.S_t[i].shape)
            elif particle.sampled_mode == 'contact':
                #res = self.step_ekf(q=q, tau=tau, dyn_sys=self.dyn_contact, x=self.x, cov=self.cov)

                #particle.mu, particle.Sigma, self.S_est[i], self.y_est[i] = self.step_ekf(q=q, tau=tau, dyn_sys=self.dyn_contact, x=self.x, cov=self.cov)

                # if particle inherits ekf
                particle.mu, particle.Sigma, self.S_est[i], self.y_est[i] = particle.step(q=q, tau=tau, dyn_sys=self.dyn_contact)


                #particle.mu = self.step(q=q, tau=tau, dyn_sys=self.dyn_contact)[0]
                #particle.Sigma = self.step(q=q, tau=tau, dyn_sys=self.dyn_contact)[1]
                #print(particle.Sigma)
                #self.S_est[i] = self.step(q=q, tau=tau, dyn_sys=self.dyn_contact)[2]
                #self.y_est[i] = self.step(q=q, tau=tau, dyn_sys=self.dyn_contact)[3]
            #print("debug1")
            print(particle.Sigma)
            #print(np.linalg.eigvalsh(particle.cov))

            #print(self.y_est[i])
            #print(np.linalg.eigvalsh(self.S_est[i]))
            #print("debug transpose")
            #print(print(self.S_est[i].T))
            #print(particle.sampled_mode)
            #print(particle.mu['q'])

    def calc_weights(self, q):
        summation = 0
        for i,particle in enumerate(self.particles):
            #print(np.linalg.eigvalsh(self.S_est[i]))
            particle.weight = multivariate_normal(mean=self.y_est[i].ravel(), cov=self.S_est[i]).pdf(q)
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
        #print(pos.shape)
        weights = np.zeros(self.num_particles)
        vel = np.zeros((self.num_particles, self.dyn_free.nq))
        #x_env = np.zeros(self.num_particles)
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

class Particle(ekf):
    def __init__(self, par, q0=np.array([2.29, -1.02, -0.9, -2.87, 1.55, 0.56]), est_geom=False, est_stiff=False):
        ekf.__init__(self, par, q0, est_geom, est_stiff)
        self.mode = self.mode_prev = par['belief_init']
        self.mu = self.x
        self.Sigma = self.cov
        self.weight = par['weight0']
        self.sampled_mode = par['sampled_mode']





