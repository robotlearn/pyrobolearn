#!/usr/bin/env python
"""Provide the continuous Gaussian action exploration strategies.

The Gaussian exploration strategy consists to explore in the continuous action space of policies by using a
gaussian distribution on action probabilities.
"""

import torch

from pyrobolearn.distributions.modules import *
from pyrobolearn.exploration.actions.continuous import ContinuousActionExploration


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GaussianActionExploration(ContinuousActionExploration):
    r"""Gaussian continuous action exploration.

    The Gaussian exploration strategy consists to explore in the continuous action space of policies by using a
    gaussian distribution on action probabilities.
    """

    def __init__(self, policy, action, module=None):
        """
        Initialize the Gaussian action exploration strategy.

        Args:
            policy (Policy): policy to wrap.
            action (Action): continuous actions.
            module (None, GaussianModule): Gaussian module.
        """
        super(GaussianActionExploration, self).__init__(policy, action=action)

        # create Gaussian module if necessary
        if module is None:
            mean = IdentityModule()
            covariance = DiagonalCovarianceModule(num_inputs=self.policy.base_output.size(-1),
                                                  num_outputs=action.size)
            module = GaussianModule(mean=mean, covariance=covariance)

        # check that the module is a Gaussian Module
        if not isinstance(module, GaussianModule):
            raise TypeError("Expecting the given 'module' to be an instance of `GaussianModule`, instead got: "
                            "{}".format(type(module)))

        self._module = module
