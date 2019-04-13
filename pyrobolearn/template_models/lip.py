#!/usr/bin/env python
"""Provide the Soft-Actor Critic algorithm.

Define the SAC reinforcement learning algorithm. This is a model-free, off-policy, actor-critic method.
"""

import numpy as np


__author__ = ["Songyan Xin", "Brian Delhaisse"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LIPM:
    """Linear Inverted Pendulum (model)."""

    def __init__(self, X0, z, g=9.81):
        """
        Initialize the LIP model.

        Args:
            X0 (np.float[2]): LIP state (position and velocity).
            z (float): height of the inverted pendulum.
            g (float): gravity in the z direction.
        """
        self.z, self.g = z, g
        self.Tc = np.sqrt(self.z / self.g)

        self.x0, self.xd0 = X0

    def __call__(self, t):
        t = np.asarray(t)
        x_t = self.x0 * np.cosh(t / self.Tc) + self.Tc * self.xd0 * np.sinh(t / self.Tc)
        xd_t = self.x0 / self.Tc * np.sinh(t / self.Tc) + self.xd0 * np.cosh(t / self.Tc)
        X_t = [x_t, xd_t]
        return np.asarray(X_t)
