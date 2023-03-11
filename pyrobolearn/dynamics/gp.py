#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the GPR dynamic transition function approximators

The GPR dynamic model predicts using a GPR model the next state given the current state and action.
"""

from pyrobolearn.approximators.gp import GPRApproximator
from pyrobolearn.dynamics.dynamic import ParametrizedDynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GPRDynamicModel(ParametrizedDynamicModel):
    r"""GPR Dynamic Model

    The GPR dynamic model predicts using a GPR model the next state given the current state and action.
    """

    def __init__(self, state, action, next_state=None, distributions=None, preprocessors=None, postprocessors=None):
        """
        Initialize the GPR dynamic transition function / probability :math:`p(s_{t+1} | s_t, a_t)`.

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
        model = GPRApproximator(inputs=[state, action], outputs=next_state, preprocessors=preprocessors,
                                postprocessors=postprocessors)
        super(GPRDynamicModel, self).__init__(state, action, model=model, next_state=next_state,
                                              distributions=distributions)
