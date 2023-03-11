#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the abstract noise class from which all noises inherit from.

The noise can be applied notably on sensors and actuators.
"""

from abc import ABCMeta

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Noise(object):
    r"""Noise (abstract) class

    Abstract Noise class.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize the noise.
        """
        pass

    def apply_noise(self, data, inplace=False):
        """
        Apply the noise on the given data.

        Args:
            data (int, float, np.array): data to apply the noise on.
            inplace (bool): if True, it will directly modify the given data, and won't return a copy of it.

        Returns:
            int, float, np.array: noisy data.
        """
        raise NotImplementedError

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self, data, inplace=False):
        """Apply the noise on the given data."""
        return self.apply_noise(data, inplace=inplace)


class NoNoise(Noise):
    r"""No noise class.

    This is a dummy class that don't apply any noises on the given data and just return it.
    """

    def apply_noise(self, data, inplace=False):
        """
        Apply the noise on the given data.

        Args:
            data (int, float, np.array): data to apply the noise on.
            inplace (bool): if True, it will directly modify the given data, and won't return a copy of it.

        Returns:
            int, float, np.array: noisy data.
        """
        return data
