# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define weights used in the forcing terms in dynamic movement primitives
"""

import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WeightTensor(torch.nn.Module):
    r"""Fixed weight vector

    Weights used with the basis functions in the forcing terms.
    """

    def __init__(self, weight=None, num_basis=None):
        """
        Initialize the fixed weight vector.

        Args:
            weight (torch.Tensor, None): weight vector. If None, it will create a weight vector using the provided
                :attr:`num_basis`.
            num_basis (int): number of basis functions. This is used if no weight vector was given.
        """
        super(WeightTensor, self).__init__()
        if weight is None:
            weight = torch.zeros(num_basis)
        elif not isinstance(weight, torch.Tensor):
            raise TypeError("Expecting the weight vector to be None or an instance of `torch.Tensor`, instead got: "
                            "{}".format(weight))
        self._weight = weight
        self._weight.requires_grad = True

    @property
    def weight(self):
        return self._weight

    def forward(self, *x):
        """Return the weight vector."""
        return self.weight


class WeightModule(torch.nn.Module):
    r"""Weight vector

    Weights used with the basis functions in the forcing terms.
    """

    def __init__(self, weight=None, num_basis=None):
        """
        Initialize the weight vector.

        Args:
            weight (torch.nn.Module, torch.Tensor, None): weight module or vector. If None, it will create a weight
                tensor/module using the provided :attr:`num_basis`.
            num_basis (int): number of basis functions. This is used if no weight vector was given.
        """
        super(WeightModule, self).__init__()
        if weight is None:
            weight = WeightTensor(num_basis=num_basis)
        elif isinstance(weight, torch.Tensor):
            weight = WeightTensor(weight=weight)
        elif not isinstance(weight, torch.nn.Module):
            raise TypeError("Expecting the weight vector to be None, an instance of `torch.Tensor`, or "
                            "`torch.nn.Module`, instead got: {}".format(type(weight)))
        self._weight = weight

    @property
    def weight(self):
        """Return a torch.Tensor or torch.nn.Module."""
        if isinstance(self._weight, WeightTensor):
            return self._weight.weight
        return self._weight

    def forward(self, x):
        """Forward the input to the weight module."""
        return self._weight(x)
