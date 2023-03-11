#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the canonical systems used in ProMPs.

This file defines the canonical systems used in ProMPs. The canonical system (CS) allows to modulate temporarily the
ProMP, that is, it provides the phase that drives the ProMP [1].

References
    - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
    - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
"""

from abc import ABCMeta
import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CS(object):
    r"""Canonical System
    """

    def step(self, tau=1., **kwargs):
        pass

    def grad(self, t=None):
        pass


class LinearCS(CS):
    r"""Linear Canonical System.

    A canonical system (CS) allows to modulate temporarily the ProMP, that is, it provides the phase that drives
    the ProMP [1].
    The phase variable was introduced to avoid an explicit dependency with time in the ProMP equations. Canonical
    systems can be categorized in two main categories:
    * discrete CS: used for discrete movements (such as reaching, pushing/pulling, hitting, etc)
    * rhythmic CS: used for rhythmic movements (such as walking, running, dribbling, sewing, flipping a pancake, etc)

    Each of these systems are described by differential equations which are solved using Euler's method.
    See their corresponding classes `DiscreteCS` and `RhythmicCS` for more information.

    References:
        - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
        - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
    """

    __metaclass__ = ABCMeta

    def __init__(self, dt=0.01, T=1.):
        """Initialize the canonical system.

        Args:
            dt (float): the time step used in Euler's method when solving the differential equation
                A very small step will lead to a better accuracy but will take more time.
            T (float): total time for the movement (period)
        """
        # set variables
        self.dt = dt
        self.T = T
        self.timesteps = int(T / self.dt)
        # rescale integration step (same as np.linspace(0.,T.,timesteps) instead of np.arange(0,T,dt))
        self.dt = self.T / (self.timesteps - 1.)

        # initial time
        self.t0 = 0.
        self.s = self.t0

        # slope
        if self.T <= 0:
            raise ValueError("Expecting the period T to be bigger than 0")
        self.slope = 1. / self.T

        # reset the phase variable
        self.reset()

    ##############
    # Properties #
    ##############

    @property
    def initial_phase(self):
        """Return the initial phase"""
        return self.t0

    @property
    def final_phase(self):
        """Return the final phase"""
        return self.T

    @property
    def num_timesteps(self):
        """Return the number of timesteps"""
        return self.timesteps

    ###########
    # Methods #
    ###########

    def reset(self):
        """
        Reset the phase variable to its initial phase.
        """
        self.s = self.t0

    def step(self, tau=1.0, error_coupling=1.0):
        """
        Perform a step using Euler's method; increment the phase by a small amount.

        Args:
            tau (float): speed. Increase tau to make the system faster, and decrease it to make it slower.

        Returns:
            float: current phase
        """
        s = self.s
        self.s += tau * self.slope * self.dt

        # return previous 's' such that it starts from t0
        return s

    # aliases
    predict = step
    forward = step

    def grad(self, t=None):
        r"""
        Compute the gradient of the phase variable with respect to time, i.e. :math:`ds/dt(t)`.

        Args:
            t (float, None): time variable

        Returns:
            float: gradient evaluated at the given time
        """
        return self.slope

    def rollout(self, tau=1.0, error_coupling=1.0):
        """
        Generate phase variable in an open loop fashion from the initial to the final phase.

        Args:
            tau (float): Increase tau to make the system faster, and decrease it to make it slower.
            error_coupling (float): slow down if the error is > 1

        Returns:
            np.array[T]: value of phase variable at each time step
        """
        timesteps = int(self.timesteps * tau)
        self.s_track = np.zeros(timesteps)

        # reset
        self.reset()

        # roll
        for t in range(timesteps):
            self.s_track[t] = self.s
            self.step(tau, error_coupling)

        return self.s_track
