# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define linear function approximator.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.basics.polynomial import Polynomial, PolynomialFunction


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PolynomialApproximator(Approximator):
    r"""Polynomial Function Approximator

    The polynomial function approximator is a discriminative deterministic model expressed mathematically as
    :math:`y = f(x) = W \phi(x)`, where :math:`x` is the input vector, :math:`y` is the output vector, :math:`W`
    is the weight matrix, and :math:`\phi` is the polynomial function which returns the transformed input vector.
    This transformed input vector is often of higher dimension, based on the idea that if it is not linear with
    respect to the parameters in the current space, it might be in a higher dimensional space.
    """

    def __init__(self, inputs, outputs, degree=1, preprocessors=None, postprocessors=None):
        """
        Initialize the polynomial approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of Action/State)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
                to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
                processors before being returned.
        """
        # create inner model
        polynomial_fct = PolynomialFunction(degree=degree)
        model = Polynomial(num_inputs=self._size(inputs), num_outputs=self._size(outputs),
                           polynomial_fct=polynomial_fct)

        # call parent class
        super(PolynomialApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                                     postprocessors=postprocessors)
