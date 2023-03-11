#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides dynamic transition neural network approximators

For instance, a dynamic network is a function represented by a neural network that maps a state-action to the next
state.
"""

from pyrobolearn.approximators import NNApproximator, MLPApproximator
from pyrobolearn.dynamics.dynamic import ParametrizedDynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NNDynamicModel(ParametrizedDynamicModel):
    r"""Neural Network Dynamic Model

    Dynamic model using neural networks.

    Pros:
    Cons: requires lot of samples, overfitting,...
    """

    def __init__(self, state, action, model, next_state=None, distributions=None, preprocessors=None,
                 postprocessors=None):
        """
        Initialize the NN dynamic model.

        Args:
            state (State): state inputs.
            action (Action): action inputs.
            next_state (State, None): state outputs. If None, it will take the state inputs as the outputs.
            model (NNApproximator, NN): neural network model.
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if model is None:
            raise TypeError("Expecting the model to be a neural network and not None.")
        elif not isinstance(model, NNApproximator):
            if next_state is None:
                next_state = state
            model = NNApproximator(inputs=[state, action], outputs=next_state, model=model,
                                   preprocessors=preprocessors, postprocessors=postprocessors)
        super(NNDynamicModel, self).__init__(state, action, model=model, next_state=next_state,
                                             distributions=distributions)


class MLPDynamicModel(NNDynamicModel):
    r"""MLP Dynamic Model

    """

    def __init__(self, state, action, next_state=None, hidden_units=(), activation='linear', last_activation=None,
                 dropout=None, distributions=None, preprocessors=None, postprocessors=None):
        """
        Initialize the multi-layer perceptron model.

        Args:
            state (State): state inputs.
            action (Action): action inputs.
            next_state (State, None): state outputs. If None, it will take the state inputs as the outputs.
            hidden_units (tuple, list of int): number of hidden units in each layer
            activation (str): activation function to apply on each layer
            last_activation (str, None): activation function to apply on the last layer
            dropout (None, float): dropout probability
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if next_state is None:
            next_state = state
        model = MLPApproximator(inputs=[state, action], outputs=next_state, hidden_units=hidden_units,
                                activation=activation, last_activation=last_activation,
                                dropout=dropout)
        super(MLPDynamicModel, self).__init__(state, action, model=model, next_state=next_state,
                                              distributions=distributions, preprocessors=preprocessors,
                                              postprocessors=postprocessors)
