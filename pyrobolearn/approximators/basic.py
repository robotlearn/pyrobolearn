#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define basic function approximators.

Define the various basic approximators such as the random approximator, linear approximator, etc.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import collections
import numpy as np
import torch

from pyrobolearn.states import State
from pyrobolearn.actions import Action

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.basics.linear import Linear

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RandomApproximator(Approximator):
    r"""Random Approximator
    """

    class Random(object):

        def __init__(self, num_outputs, seed=None):
            self.num_outputs = num_outputs
            if seed is not None:
                np.random.seed(seed)

    def __init__(self, outputs, preprocessors=None, postprocessors=None):
        """
        Initialize the random approximator.

        Args:
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """
        # call parent class
        model = self.Random(num_outputs=self._size(outputs), seed=None)
        super(RandomApproximator, self).__init__(inputs=None, outputs=outputs, model=model,
                                                 preprocessors=preprocessors, postprocessors=postprocessors)


class LinearApproximator(Approximator):
    r"""Linear Function Approximator
    """

    def __init__(self, inputs, outputs, preprocessors=None, postprocessors=None):
        """
        Initialize the linear approximator.

        Args:
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """
        # call parent class
        model = Linear(num_inputs=self._size(inputs), num_outputs=self._size(outputs), add_bias=True)
        super(LinearApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                                 postprocessors=postprocessors)
