#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define Kernel Movement Primitive (KMP) function approximator.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.kmp import KMP


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KMPApproximator(Approximator):
    r"""Kernel Movement Primitive Approximator

    """

    def __init__(self, inputs, outputs, num_components=1, priors=None, means=None, covariances=None, gaussians=None,
                 preprocessors=None, postprocessors=None):
        """
        Initialize the Kernel Movement Primitive approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of Action/State)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            num_components (int): the number of components/gaussians (this argument should be provided if
              no priors, means, covariances, or gaussians are provided)
            priors (list/tuple of float, None): prior probabilities (they have to be positives). If not provided,
              it will be a uniform distribution.
            means (list of np.array[float[D]], None): list of means
            covariances (list of np.array[float[D,D]], None): list of covariances
            gaussians (list of Gaussian, None): list of gaussians. If provided, the `means` and `covariances`
              parameters don't have to be provided.
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
              to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
              processors before being returned.
        """
        # create inner model
        num_inputs, num_outputs = self._size(inputs), self._size(outputs)
        dimensionality = num_inputs + num_outputs
        model = KMP()

        # call parent class
        super(KMPApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                              postprocessors=postprocessors)
