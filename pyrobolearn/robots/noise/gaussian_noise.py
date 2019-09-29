# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the Gaussian noise class.

This noise can be applied notably on sensors and actuators.
"""

import numpy as np

from pyrobolearn.robots.noise.noise import Noise

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GaussianNoise(Noise):
    r"""Gaussian noise class

    This computes:

    .. math:: y = x + \epsilon

    where :math:`x` is the original data, :math:`y` is the noisy data, and
    :math:`\epsilon \sim \mathcal{N}(\mu, \Sigma)` is the added Gaussian noise.
    """

    def __init__(self, vars=1):
        """Initialize the noise.

        Args:
            vars (float, int, np.array): variances or covariance matrix.
        """
        super(GaussianNoise, self).__init__()
        self._multivariate = True if isinstance(vars, np.ndarray) and vars.ndim == 2 else False
        if self._multivariate:
            mean = np.zeros(vars.shape[0])
        else:
            if not isinstance(vars, (int, float, np.ndarray)):
                raise TypeError("Expecting the given 'vars' to be an int, float, or np.array, instead got: "
                                "{}".format(type(vars)))
            if isinstance(vars, np.ndarray) and vars.ndim > 1:
                raise ValueError("Expecting the given 'vars' to be a np.array of dim 1 or 2, instead got: "
                                 "{}".format(vars.ndim))
            vars = np.sqrt(vars)  # abuse of name; this is no more the variance but the standard deviation
            mean = 0.
        self.vars = vars
        self.mean = mean

    def apply_noise(self, data, inplace=False):
        """
        Apply the noise on the given data.

        Args:
            data (int, float, np.array): data to apply the noise on.
            inplace (bool): if True, it will directly modify the given data, and won't return a copy of it.

        Returns:
            int, float, np.array: noisy data.
        """
        # compute noise
        if self._multivariate:
            if isinstance(data, np.ndarray) and data.shape != self.vars.shape:
                noise = np.random.multivariate_normal(mean=self.mean, cov=self.vars, size=data.shape)
            else:
                noise = np.random.multivariate_normal(mean=self.mean, cov=self.vars)
        else:
            if isinstance(data, np.ndarray):
                noise = np.random.normal(loc=self.mean, scale=self.vars, size=data.shape)
            else:
                noise = np.random.normal(loc=self.mean, scale=self.vars)

        # check if inplace operation
        if not inplace:
            return data + noise
        data += noise
        return data

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__ + "(mean=0, vars=" + str(self.vars) + ")"
