# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the continuous Gaussian parameter exploration strategies.

The Gaussian parameter exploration strategy consists to explore in the parameter space of policies by using a gaussian
distribution on the parameters.
"""

import torch

# from pyrobolearn.distributions.modules import FixedMeanModule, FixedCovarianceModule, GaussianModule
from pyrobolearn.distributions.gaussian import Gaussian
from pyrobolearn.exploration.parameters.parameter_exploration import ParameterExploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GaussianParameterExploration(ParameterExploration):
    r"""Gaussian continuous parameter exploration.

    The Gaussian exploration strategy consists to explore in the continuous parameter space of policies by using a
    gaussian distribution on parameters.
    """

    def __init__(self, policy, variance=1., module=None):
        """
        Initialize the Gaussian action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            variance (float): variance parameter for the covariance.
            module (None, GaussianModule): Gaussian module.
        """
        super(GaussianParameterExploration, self).__init__(policy)

        # create Gaussian module if necessary
        if module is None:
            mean = torch.tensor(self.parameters, requires_grad=True)
            # mean = FixedMeanModule(mean=mean)
            covariance = variance * torch.eye(self.size, requires_grad=True)
            # covariance = FixedCovarianceModule(covariance=covariance)
            # module = GaussianModule(mean=mean, covariance=covariance)
            module = Gaussian(mean=mean, covariance=covariance)

        # check that the module is a Gaussian Module
        if not isinstance(module, (torch.distributions.Normal, torch.distributions.MultivariateNormal)):
            raise TypeError("Expecting the given 'module' to be an instance of `torch.distributions.Normal` or "
                            "`torch.distributions.MultivariateNormal`, instead got: {}".format(type(module)))

        self._module = module

    ##############
    # Properties #
    ##############

    @property
    def module(self):
        """Return the module instance."""
        return self._module

    ###########
    # Methods #
    ###########

    def sample(self):
        """Sample the parameters from the """
        parameters = self.module.rsample((1,))  # rsample allows to get the gradients
        return parameters

