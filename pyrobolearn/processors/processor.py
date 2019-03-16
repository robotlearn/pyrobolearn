#!/usr/bin/env python
"""Define the Processor class.

Processors are rules that are applied to the inputs and outputs of a learning model before being processed by the
model or after. Processors might have parameters but they do not have trainable/optimizable parameters; the parameters
are fixed and given at the beginning.
"""

import numpy as np
import torch


class Processor(object):
    r"""Processor

    Processors are rules that are applied to the inputs and outputs of a model before being processed by the model
    or after. Processors might have parameters but they do not have trainable/optimizable parameters; the parameters
    are fixed and given at the beginning.
    """

    def __init__(self):
        pass

    def compute(self, x):
        pass

    def __call__(self, x):
        return self.compute(x)


class CenterProcessor(Processor):
    r"""Center Processor

    Center the data by the given mean; that is, it returned: :math:`\hat{x} = x - \mu` where :math:`\mu` is the mean.
    """

    def __init__(self, mean):
        super(CenterProcessor, self).__init__()
        self.mean = torch.Tensor(mean)

    def compute(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            x -= self.mean
            return x.numpy()
        return x - self.mean


class StandardizerProcessor(Processor):
    r"""Standardizer Processor

    Processor that standardize the given data; the returned data is centered around 0 with a standard deviation of 1.
    That is, it returned :math:`\hat{x} = \frac{x - \mu}{\sigma}`, where :math:`\mu` is the mean, and :math:`\sigma`
    is the standard deviation.
    """

    def __init__(self, mean, std):
        super(StandardizerProcessor, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def compute(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            x = (x - self.mean) / (self.std + 1.e-13)
            return x.numpy()
        return (x - self.mean) / (self.std + 1.e-13)


class NormalizerProcessor(Processor):
    r"""Normalizer Processor

    Processor that normalize the given data; the returned data will be between 0 and 1.
    That is, it returned :math:`\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}`, where
    :math:`x \in [x_{min}, x_{max}]`.
    """

    def __init__(self, xmin, xmax):
        super(NormalizerProcessor, self).__init__()
        self.xmin = torch.Tensor(xmin)
        self.xmax = torch.Tensor(xmax)
        if torch.allclose(self.xmin, self.xmax):
            raise ValueError("The given arguments 'xmin' and 'xmax' are the same.")

    def compute(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            x = (x - self.xmin) / (self.xmax - self.xmin)
            return x.numpy()
        return (x - self.xmin) / (self.xmax - self.xmin)
