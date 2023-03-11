#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define linear function approximator.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.basics.linear import Linear


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearApproximator(Approximator):
    r"""Linear Function Approximator

    The linear function approximator is a linear parametric model: :math:`y = W x + b` where :math:`x` and :math:`y`
    are respectively the input and output vectors, :math:`W` is the weight matrix, and :math:`b` is the bias/intercept.
    """

    def __init__(self, inputs, outputs, preprocessors=None, postprocessors=None):
        """
        Initialize the linear approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of Action/State)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """
        # create inner model
        model = Linear(num_inputs=self._size(inputs), num_outputs=self._size(outputs), add_bias=True)

        # call parent class
        super(LinearApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                                 postprocessors=postprocessors)
