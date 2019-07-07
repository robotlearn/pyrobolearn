#!/usr/bin/env python
"""Provides the polynomial dynamic transition function approximators

The polynomial dynamic model predicts using a polynomial model the next state given the current state and action.
"""

from pyrobolearn.approximators import PolynomialApproximator
from pyrobolearn.dynamics.dynamic import ParametrizedDynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PolynomialDynamicModel(ParametrizedDynamicModel):
    r"""Polynomial Dynamic Model

    The polynomial dynamic model predicts using a polynomial model the next state given the current state and action.
    """

    def __init__(self, state, action, next_state=None, distributions=None, degree=1, preprocessors=None,
                 postprocessors=None):
        """
        Initialize the polynomial dynamic transition function / probability :math:`p(s_{t+1} | s_t, a_t)`.

        Args:
            state (State): state inputs.
            action (Action): action inputs.
            next_state (State, None): state outputs. If None, it will take the state inputs as the outputs.
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if next_state is None:
            next_state = state
        model = PolynomialApproximator(inputs=[state, action], outputs=next_state, degree=degree,
                                       preprocessors=preprocessors, postprocessors=postprocessors)
        super(PolynomialDynamicModel, self).__init__(state, action, model=model, next_state=next_state,
                                                     distributions=distributions)
