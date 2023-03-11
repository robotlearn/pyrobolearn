#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Histogram Filter.
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class HistogramFilter(object):
    r"""Histogram Filter (HF)

    Type: Non-parametric filter

    The histogram filter decomposes the continuous state space into finite regions, and represents the posterior
    by a histogram. [1]

    Notes:
        - this filter is well-suited to represent complex multimodal belief [1]
        - the complexity depends on the number of parameters

    Complexity: :math:`O(M^N)` where :math:`M` is the number of regions/bins, and :math:`N` is the dimensionality
        of the state.

    References:
        [1] "Probabilistic Robotics", Thrun et al., 2006 (sec 4.1)
    """

    def __init__(self, state_dim=1, num_bins_per_dim=10):
        self.num_bins = num_bins_per_dim
        self.state_dim = state_dim

        # total number of parameters (=regions)
        self.num_bins = num_bins_per_dim**state_dim

        # initial belief: uniform distribution
        p = 1./self.num_bins
        self.p = np.full([num_bins_per_dim]*state_dim, p)

    def predict(self, f, u):
        """
        Predict (a priori) the next state (without taking into account measurements) using the probabilistic
        nonlinear dynamic model.

        Args:
            f (callable class/function): probabilistic (nonlinear) dynamical function that stacks on the axis 0 the
                probability distribution for each region. That is the size of axis 0 should be
                :math:`num(bins)^{dim(state)}`.
            u (np.array): control array
        """
        # probability to arrive in region k given control input u from any region i (convolution operation)
        s = ''.join([chr(i) for i in range(97, 97 + self.state_dim + 1)])
        self.p = np.einsum(s+','+s[1:]+'->'+s[1:], f(self.p.shape, u), self.p)

        # return belief
        return self.p

    def measurement_update(self, h, z):
        """
        Predict (a posteriori) the next state by incorporating the measurement :math:`z_t`.

        Args:
            h (callable class/function): probabilistic (nonlinear) measurement function to see measurement from
                region k. This function should return an array of the same shape as the one provided in argument
                where each cell contains the probability to see the measurement `z` from that region (i.e. index).
            z (np.array): measurement array
        """
        # probability to see measurement z from region k (multiplication operation)
        self.p = h(z, self.p.shape) * self.p

        # normalize to get a proper probability distribution
        self.p /= np.sum(self.p)

        # return belief
        return self.p

    def compute(self, f, u, h, z):
        """
        Perform a prediction and measurement update step.

        Args:
            f (callable class/function): probabilistic (nonlinear) dynamical function that stacks on the axis 0 the
                probability distribution for each region. That is the size of axis 0 should be
                :math:`num(bins)^{dim(state)}`.
            u (np.array): control array
            h (callable class/function):  probabilistic (nonlinear) measurement function to see measurement from
                region k. This function should return an array of the same shape as the one provided in argument
                where each cell contains the probability to see the measurement `z` from that region (i.e. index).
            z (np.array): measurement array
        """
        self.predict(f, u)
        self.measurement_update(h, z)
        return self.p
