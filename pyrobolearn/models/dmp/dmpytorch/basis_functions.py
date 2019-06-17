#!/usr/bin/env python
"""Define basis functions used in the forcing terms in dynamic movement primitives

This file implements basis functions used for discrete and rhythmic dynamic movement primitives.
"""

from abc import ABCMeta, abstractmethod
import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BF(torch.nn.Module):
    r"""Basis function used in the forcing terms
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BF, self).__init__()


class EBF(BF):
    r"""Exponential basis function

    This basis function is given by the formula:

    .. math:: \psi(s) = \exp \left( - \frac{1}{2 \sigma^2} (s - c)^2 \right)

    where :math:`c` is the center, and :math:`\sigma` is the width of a normal distribution.

    This is often used for discrete DMPs.
    """
    def __init__(self, center=0, sigma=1., h=None):
        """Initialize basis function

        Args:
            center (float, torch.Tensor): center of the distribution
            sigma (float, torch.Tensor): width of the distribution
            h (float, torch.Tensor): concentration/precision of the basis fct (h = 1/(2*\sigma^2)).
                                   if h is not provided, it will check sigma.
        """
        super(EBF, self).__init__()

        if isinstance(center, torch.Tensor):
            pass

        self.c = center
        if h is None:
            self.h = 1. / (2*sigma**2)  # measure the concentration
        else:
            self.h = h

    def forward(self, s):
        if isinstance(s, torch.Tensor):
            s = s[:, None]
        return torch.exp(-self.h * (s - self.c)**2)


class CBF(BF):
    r"""Circular basis function (aka von Mises basis function)

    This basis function is given by the formula:

    .. math:: \psi(s) = \exp \left( h (\cos(s - c) - 1) \right)

    where :math:`c` is the center, and :math:`h` is a measure of concentration.

    This is often used for rhythmic DMPs.
    """

    def __init__(self, center=0, h=1.):
        """Initialize basis function

        Args:
            center (float, torch.Tensor): center of the basis fct
            h (float, torch.Tensor): concentration/precision of the basis fct
        """
        super(CBF, self).__init__()
        self.c = center
        self.h = h

    def forward(self, s):
        if isinstance(s, torch.Tensor):
            s = s[:, None]
        # return torch.exp(self.h * torch.cos(s - self.c) - 1)    # this is bad as it is not bounded as we increase
                                                                  # the number of basis functions.
        return torch.exp(self.h * torch.cos(s - self.c) - self.h)
