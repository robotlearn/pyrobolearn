#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Uniform noise class.

This noise can be applied notably on sensors and actuators.
"""

import numpy as np

from pyrobolearn.robots.noise.noise import Noise

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UniformNoise(Noise):
    r"""Uniform noise class

    This computes:

    .. math:: y = x + \epsilon

    where :math:`x` is the original data, :math:`y` is the noisy data, and
    :math:`\epsilon \sim \mathcal{U}(a=-\frac{w}{2}, b=\frac{w}{2})` is the added Uniform noise where :math:`w = b-a`
    is the width of the distribution.
    """

    def __init__(self, width=1.):
        """Initialize the noise.

        Args:
            width (float, int, np.array): variances or covariance matrix.
        """
        super(UniformNoise, self).__init__()
        self.width = width
        self.radius = self.width/2.

    def apply_noise(self, data, inplace=False):
        """
        Apply the noise on the given data.

        Args:
            data (int, float, np.array): data to apply the noise on.
            inplace (bool): if True, it will directly modify the given data, and won't return a copy of it.

        Returns:
            int, float, np.array: noisy data.
        """
        if isinstance(data, np.ndarray):
            noise = np.random.uniform(low=-self.radius, high=self.radius, size=data.shape)
        else:
            noise = np.random.uniform(low=-self.radius, high=self.radius)

        # check if inplace operation
        if not inplace:
            return data + noise
        data += noise
        return data

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__ + "(a=" + str(-self.radius) + "b=" + str(self.radius) + ")"
