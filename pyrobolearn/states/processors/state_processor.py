#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide state processors

The various state processors defined here allow to process the data after being read. In contrast, to process the
data for each approximators (policy, value function, etc), you can process the state once here. However, it might
not always be beneficial to process it; other part of the code might require the real original state and not the
processed one.
"""

# TODO: correct the few mistakes!

import numpy as np

from pyrobolearn.states.state import State

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class StateProcessor(State):
    r"""State Processor

    The state processor allows to pre-process the data by first reading it and then process it.
    """

    def __init__(self, state):
        """
        Initialize the state processor.

        Args:
            state (State): state to process.
        """
        super(StateProcessor, self).__init__()
        self.state = state

    def read(self):
        data = self.state()
        return data


class ShiftStateProcessor(StateProcessor):
    r"""Shift State Processor

    Shift the data by the given amount; that is, it returned: :math:`\hat{x} = x + z` where :math:`z` is the
    specified amount to shift the original input :math:`x`.
    """

    def __init__(self, state, z):
        """
        Initialize the Shift Processor.

        Args:
            state (State): state to process.
            z (int, float, np.array, torch.Tensor): amount to be shifted.
        """
        super(ShiftStateProcessor, self).__init__(state)
        if isinstance(z, (int, float)):
            z = [z]
        self.z = np.array(z)

    def read(self):
        data = self.state()
        return data - self.z


class RunningCenterStateProcessor(StateProcessor):
    r"""Running Center State Processor

    Center the data by using the mean which is updated each time a new data point is given.
    """

    def __init__(self, state):
        super(RunningCenterStateProcessor, self).__init__(state)
        self.mean = 0.
        self.N = 0

    def reset(self):
        self.mean = 0.
        self.N = 0

    def read(self):
        x = self.state()
        # update the mean
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * x
        self.N += 1

        # center the data with new mean
        return x - self.mean


class StandardizerStateProcessor(StateProcessor):
    r"""Standardizer State Processor

    Processor that standardizes the given data; the returned data is centered around 0 with a standard deviation of 1.
    That is, it returned :math:`\hat{x} = \frac{x - \mu}{\sigma}`, where :math:`\mu` is the mean, and :math:`\sigma`
    is the standard deviation.
    """

    def __init__(self, state, mean=0., std_dev=1., epsilon=1.e-4):
        """
        Initialize the Standardizer Processor.

        Args:
            state (State): state to process.
            mean (int, float, np.array, torch.Tensor): mean
            std_dev (int, float, np.array, torch.Tensor): standard deviation
            epsilon (float): small number to be added to the denominator for stability in case the std dev = 0
        """
        super(StandardizerStateProcessor, self).__init__(state)
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std_dev, (int, float)):
            std_dev = [std_dev]
        self.mean = np.array(mean)
        self.std = np.array(std_dev)
        self.eps = epsilon

    def read(self):
        x = self.state()
        return (x - self.mean) / (self.std + self.eps)


class RunningStandardizerStateProcessor(StateProcessor):
    r"""Running Standardizer State Processor

    Processor that standardizes the given data; the returned data is centered around 0 with a standard deviation of 1.
    That is, it returned :math:`\hat{x} = \frac{x - \mu}{\sigma}`, where :math:`\mu` is the mean, and :math:`\sigma`
    is the standard deviation. The mean and the standard deviation (or variance) are updated at each time a new data
    point is given.
    """

    def __init__(self, state, epsilon=1.e-4):
        """
        Initialize the Running Standardizer Processor.

        Args:
            state (State): state to process.
            epsilon (float): small number to be added to the denominator for stability in case the std dev = 0
        """
        super(RunningStandardizerStateProcessor, self).__init__(state)
        self.mean = 0.
        self.var = 1.
        self.N = 0
        self.eps = epsilon

    def reset(self):
        self.mean = 0.
        self.var = 1.
        self.N = 0

    def read(self):
        x = self.state()
        # update the mean
        old_mean = np.copy(self.mean)
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * x

        # update the var / stddev
        self.var = self.N / (self.N + 1) * self.var + 1. / (self.N + 1) * (x - old_mean) * (x - self.mean)
        std = np.sqrt(self.var)

        # update total number of data points
        self.N += 1

        # standardize the data
        return (x - self.mean) / (std + self.eps)


class NormalizerStateProcessor(StateProcessor):
    r"""Normalizer State Processor

    Processor that normalizes the given data; the returned data will be between 0 and 1.
    That is, it returned :math:`\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}`, where
    :math:`x \in [x_{min}, x_{max}]`.
    """

    def __init__(self, state, xmin, xmax):
        """
        Initialize the Normalizer Processor.

        Args:
            state (State): state to process.
            xmin (int, float, np.array, torch.Tensor): minimum bound
            xmax (int, float, np.array, torch.Tensor): maximum bound
        """
        super(NormalizerStateProcessor, self).__init__(state)
        if isinstance(xmin, (int, float)):
            xmin = [xmin]
        if isinstance(xmax, (int, float)):
            xmax = [xmax]
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        if torch.allclose(self.xmin, self.xmax):
            raise ValueError("The given arguments 'xmin' and 'xmax' are the same.")

    def read(self):
        x = self.state()
        return (x - self.xmin) / (self.xmax - self.xmin)


class RunningNormalizerStateProcessor(StateProcessor):
    r"""Running Normalizer State Processor

    Processor that normalizes the given data; the returned data will be between 0 and 1.
    That is, it returned :math:`\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}`, where
    :math:`x \in [x_{min}, x_{max}]`. The :math:`x_{min}` and `x_{max}` will be updated each time a new data point
    is given.

    Warnings: it will return zero at the beginning as x = x_min = x_max.
    """

    def __init__(self, state):
        super(RunningNormalizerStateProcessor, self).__init__(state)
        self.xmin = 0.
        self.xmax = 1.

    def reset(self):
        self.xmin = 0.
        self.xmax = 1.

    def read(self):
        x = self.state()
        # compute new xmin and xmax given new data point
        self.xmin = np.minimum(x, self.xmin)
        self.xmax = np.maximum(x, self.xmax)

        # if xmax and xmin are not different, make the denominator to be 1
        idx = (self.xmax == self.xmin)
        self.xmax[idx] = self.xmin[idx] + 1.

        # normalize
        return (x - self.xmin) / (self.xmax - self.xmin)


class ClipStateProcessor(StateProcessor):
    r"""Clip State Processor

    Processor that clips the given data; the returned data will be between [low, high], where `low` and `high` are
    respectively the specified lower and higher bound.
    """

    def __init__(self, state, low=-10., high=10.):
        """
        Initialize the Clip processor.

        Args:
            state (State): state to process.
            low (int, float, np.array, torch.Tensor): lower bound
            high (int, float, np.array, torch.Tensor): higher bound
        """
        super(ClipStateProcessor, self).__init__(state)
        if isinstance(low, (int, float)):
            low = [low]
        if isinstance(high, (int, float)):
            high = [high]
        self.low = np.array(low)
        self.high = np.array(high)

    def read(self):
        x = self.state()
        return np.clip(x, self.low, self.high)


class ScaleStateProcessor(StateProcessor):
    r"""Scale processor

    Processor that scales the input x which is between [x1, x2] to the output y which is between [y1, y2].
    This is for instance useful after a tanh layer in a neural network which outputs a value between -1 and 1, and
    that value has to be rescaled to a bigger (absolute) value.
    """

    def __init__(self, state, x1, x2, y1, y2):
        """
        Initialize the scale processor.

        Args:
            state (State): state to process.
            x1 (int, float, np.array, torch.Tensor): lower bound of input
            x2 (int, float, np.array, torch.Tensor): upper bound of input
            y1 (int, float, np.array, torch.Tensor): lower bound of output
            y2 (int, float, np.array, torch.Tensor): upper bound of output
        """

        super(ScaleStateProcessor, self).__init__(state)

        def convert(x):
            if isinstance(x, (int, float)):
                return [x]
            return x

        self.x1 = np.array(convert(x1))
        self.x2 = np.array(convert(x2))
        self.y1 = np.array(convert(y1))
        self.y2 = np.array(convert(y2))
        self.ratio = (self.y2 - self.y1) / (self.x2 - self.x1)

    def read(self):
        x = self.state()
        return self.y1 + (x - self.x1) * self.ratio
