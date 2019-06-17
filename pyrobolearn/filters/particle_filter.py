#!/usr/bin/env python
"""Define the Particle Filter.
"""

import numpy as np
import bisect

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Particle(object):
    r"""Particle
    """
    def __init__(self, state):
        self.state = state


class ParticleFilter(object):
    r"""Particle Filter

    Type: Non-parametric filter

    "Particle filtering uses a genetic mutation-selection sampling approach, with a set of particles
    (also called samples) to represent the posterior distribution of some stochastic process given noisy
    and/or partial observations." (Wikipedia)
    In other words, "a particle is a hypothesis as to what the true world state may be at time t." [1]

    Complexity: O(exp(n))

    References:
        [1] "Probabilistic Robotics", Thrun et al., 2006 (sec 4.3)
    """

    def __init__(self, state_min, state_max, num_particles=100):
        # initialize particles randomly between the 2 given bounds (i.e. state_min and state_max)
        self.particles = np.random.uniform(state_min, state_max, size=(num_particles, state_min.size))
        self.weights = np.ones(num_particles)

    @property
    def num_particles(self):
        return len(self.particles)

    def predict(self, f, u):
        # for each particle, predict next state of the particle given the control input
        self.particles = [f(state, u) for state in self.particles]
        return self.particles

    def measurement_update(self, h, z):
        # for each particle, compute the probability to see measurement z from the particle state
        self.weights = [h(z, state) for state in self.particles]
        return self.weights

    def sampling(self):
        # importance sampling (survival of the fittest)
        particles = []

        # resampling

        # Wheel algorithm (from Udacity)
        # idx = np.random.randint(0, self.num_particles)
        # b, wmax = 0., max(self.weights)
        # for i in range(self.num_particles):
        #     b += np.random.random() * 2 * wmax
        #     while self.weights[idx] < b:
        #         b -= self.weights[idx]
        #         idx = (idx+1) % self.num_particles
        #     particles.append(self.particles[idx])

        # resampling O(N*log(N)) algorithm: I observe that this one was better than the Wheel algorithm
        # compute cumulative probability
        sumProb = sum(self.weights)
        cumulativeProb = [w/sumProb for w in self.weights]      # normalize
        for i in range(1, self.num_particles):
            cumulativeProb[i] += cumulativeProb[i-1]

        # resample
        for i in range(self.num_particles):
            idx = bisect.bisect(cumulativeProb, np.random.uniform(0,1))
            particles.append(self.particles[idx])

        # return particles (=states)
        self.particles = particles
        return self.particles

    def compute(self, f, u, h, z):
        self.predict(f, u)
        self.measurement_update(h, z)
        self.sampling()
        return self.particles
