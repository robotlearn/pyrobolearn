#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide basic processor classes that center, normalize, or standardize the given input.

Processors are functions that are applied to the inputs (respectively outputs) of an approximator/learning model
before (respectively after) being processed by it. Processors might have parameters but they do not have
trainable/optimizable parameters; the parameters are fixed and given at the beginning.
"""

import torch

from pyrobolearn.processors.processor import Processor, convert_numpy

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: update the methods such that x can be a matrix of data points (2d tensor), and not only one data point
#  (1d tensor)

class ShiftProcessor(Processor):
    r"""Shift Processor

    Shift the data by the given amount; that is, it returned: :math:`\hat{x} = x + z` where :math:`z` is the
    specified amount to shift the original input :math:`x`.
    """

    def __init__(self, z):
        """
        Initialize the Shift Processor.

        Args:
            z (int, float, np.array, torch.Tensor): amount to be shifted.
        """
        super(ShiftProcessor, self).__init__()
        if isinstance(z, (int, float)):
            z = [z]
        self.z = torch.tensor(z, dtype=torch.float)

    @convert_numpy
    def compute(self, x):
        return x - self.z


class RunningCenterProcessor(Processor):
    r"""Running Center Processor

    Center the data by using the mean which is updated each time a new data point is given.
    """

    def __init__(self):
        super(RunningCenterProcessor, self).__init__()
        self.mean = torch.zeros(1)
        self.N = 0

    def reset(self):
        self.mean = torch.zeros(1)
        self.N = 0

    @convert_numpy
    def compute(self, x):
        # update the mean
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * x
        self.N += 1

        # center the data with new mean
        return x - self.mean


class StandardizerProcessor(Processor):
    r"""Standardizer Processor

    Processor that standardizes the given data; the returned data is centered around 0 with a standard deviation of 1.
    That is, it returned :math:`\hat{x} = \frac{x - \mu}{\sigma}`, where :math:`\mu` is the mean, and :math:`\sigma`
    is the standard deviation.
    """

    def __init__(self, mean=0., std_dev=1., epsilon=1.e-4):
        """
        Initialize the Standardizer Processor.

        Args:
            mean (int, float, np.array, torch.Tensor): mean
            std_dev (int, float, np.array, torch.Tensor): standard deviation
            epsilon (float): small number to be added to the denominator for stability in case the std dev = 0
        """
        super(StandardizerProcessor, self).__init__()
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std_dev, (int, float)):
            std_dev = [std_dev]
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std_dev, dtype=torch.float)
        self.eps = epsilon

    @convert_numpy
    def compute(self, x):
        return (x - self.mean) / (self.std + self.eps)


class RunningStandardizerProcessor(Processor):
    r"""Running Standardizer Processor

    Processor that standardizes the given data; the returned data is centered around 0 with a standard deviation of 1.
    That is, it returned :math:`\hat{x} = \frac{x - \mu}{\sigma}`, where :math:`\mu` is the mean, and :math:`\sigma`
    is the standard deviation. The mean and the standard deviation (or variance) are updated at each time a new data
    point is given.
    """

    def __init__(self, epsilon=1.e-4):
        """
        Initialize the Running Standardizer Processor.

        Args:
            epsilon (float): small number to be added to the denominator for stability in case the std dev = 0
        """
        super(RunningStandardizerProcessor, self).__init__()
        self.mean = torch.zeros(1)
        self.var = torch.ones(1)
        self.N = 0
        self.eps = epsilon

    def reset(self):
        self.mean = torch.zeros(1)
        self.var = torch.ones(1)
        self.N = 0

    @convert_numpy
    def compute(self, x):
        # update the mean
        old_mean = torch.clone(self.mean)
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * x

        # update the var / stddev
        self.var = self.N / (self.N + 1) * self.var + 1. / (self.N + 1) * (x - old_mean) * (x - self.mean)
        std = torch.sqrt(self.var)

        # update total number of data points
        self.N += 1

        # standardize the data
        return (x - self.mean) / (std + self.eps)


class NormalizerProcessor(Processor):
    r"""Normalizer Processor

    Processor that normalizes the given data; the returned data will be between 0 and 1.
    That is, it returned :math:`\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}`, where
    :math:`x \in [x_{min}, x_{max}]`.
    """

    def __init__(self, xmin, xmax):
        """
        Initialize the Normalizer Processor.

        Args:
            xmin (int, float, np.array, torch.Tensor): minimum bound
            xmax (int, float, np.array, torch.Tensor): maximum bound
        """
        super(NormalizerProcessor, self).__init__()
        if isinstance(xmin, (int, float)):
            xmin = [xmin]
        if isinstance(xmax, (int, float)):
            xmax = [xmax]
        self.xmin = torch.tensor(xmin, dtype=torch.float)
        self.xmax = torch.tensor(xmax, dtype=torch.float)
        if torch.allclose(self.xmin, self.xmax):
            raise ValueError("The given arguments 'xmin' and 'xmax' are the same.")

    @convert_numpy
    def compute(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin)


class RunningNormalizerProcessor(Processor):
    r"""Running Normalizer Processor

    Processor that normalizes the given data; the returned data will be between 0 and 1.
    That is, it returned :math:`\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}`, where
    :math:`x \in [x_{min}, x_{max}]`. The :math:`x_{min}` and `x_{max}` will be updated each time a new data point
    is given.

    Warnings: it will return zero at the beginning as x = x_min = x_max.
    """

    def __init__(self):
        super(RunningNormalizerProcessor, self).__init__()
        self.xmin = torch.zeros(1)
        self.xmax = torch.ones(1)

    def reset(self):
        self.xmin = torch.zeros(1)
        self.xmax = torch.ones(1)

    @convert_numpy
    def compute(self, x):
        # compute new xmin and xmax given new data point
        self.xmin = torch.min(x, self.xmin)
        self.xmax = torch.max(x, self.xmax)

        # if xmax and xmin are not different, make the denominator to be 1
        idx = (self.xmax == self.xmin)
        self.xmax[idx] = self.xmin[idx] + 1.

        # normalize
        return (x - self.xmin) / (self.xmax - self.xmin)


class ClipProcessor(Processor):
    r"""Clip Processor

    Processor that clips the given data; the returned data will be between [low, high], where `low` and `high` are
    respectively the specified lower and higher bound.
    """

    def __init__(self, low=-10., high=10.):
        """
        Initialize the Clip processor.

        Args:
            low (int, float, np.array, torch.Tensor): lower bound
            high (int, float, np.array, torch.Tensor): higher bound
        """
        super(ClipProcessor, self).__init__()
        if isinstance(low, (int, float)):
            low = [low]
        if isinstance(high, (int, float)):
            high = [high]
        self.low = torch.tensor(low, dtype=torch.float)
        self.high = torch.tensor(high, dtype=torch.float)

    @convert_numpy
    def compute(self, x):
        return torch.min(torch.max(x, self.low), self.high)


class ScaleProcessor(Processor):
    r"""Scale processor

    Processor that scales the input x which is between [x1, x2] to the output y which is between [y1, y2].
    This is for instance useful after a tanh layer in a neural network which outputs a value between -1 and 1, and
    that value has to be rescaled to a bigger (absolute) value.
    """

    def __init__(self, x1, x2, y1, y2):
        """
        Initialize the scale processor.

        Args:
            x1 (int, float, np.array, torch.Tensor): lower bound of input
            x2 (int, float, np.array, torch.Tensor): upper bound of input
            y1 (int, float, np.array, torch.Tensor): lower bound of output
            y2 (int, float, np.array, torch.Tensor): upper bound of output
        """

        super(ScaleProcessor, self).__init__()

        def convert(x):
            if isinstance(x, (int, float)):
                return [x]
            return x

        self.x1 = torch.tensor(convert(x1), dtype=torch.float)
        self.x2 = torch.tensor(convert(x2), dtype=torch.float)
        self.y1 = torch.tensor(convert(y1), dtype=torch.float)
        self.y2 = torch.tensor(convert(y2), dtype=torch.float)
        self.ratio = (self.y2 - self.y1) / (self.x2 - self.x1)

    @convert_numpy
    def compute(self, x):
        return self.y1 + (x - self.x1) * self.ratio
