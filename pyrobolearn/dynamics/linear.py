#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the linear dynamic transition function approximators

The linear dynamic model predicts using a linear model the next state given the current state and action.
"""

from pyrobolearn.approximators import LinearApproximator
from pyrobolearn.dynamics.dynamic import ParametrizedDynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearDynamicModel(ParametrizedDynamicModel):
    r"""Linear Dynamic Model

    The linear dynamic model predicts using a linear model the next state given the current state and action.

    Pros: easy to implement and learn
    Cons: very limited
    """

    def __init__(self, state, action, next_state=None, distributions=None, preprocessors=None, postprocessors=None):
        """
        Initialize the linear dynamic transition function / probability :math:`p(s_{t+1} | s_t, a_t)`.

        Args:
            state (State): state inputs.
            action (Action): action inputs.
            next_state (State, None): state outputs. If None, it will take the state inputs as the outputs.
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if next_state is None:
            next_state = state
        model = LinearApproximator(inputs=[state, action], outputs=next_state, preprocessors=preprocessors,
                                   postprocessors=postprocessors)
        super(LinearDynamicModel, self).__init__(state, action, model=model, next_state=next_state,
                                                 distributions=distributions)
