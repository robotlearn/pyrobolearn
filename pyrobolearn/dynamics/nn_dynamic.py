#!/usr/bin/env python
"""Provides dynamic transition neural network approximators

For instance, a dynamic network is a function represented by a neural network that maps a state-action to the next
state.
"""

from pyrobolearn.approximators import NNApproximator, MLPApproximator
from pyrobolearn.dynamics.dynamic import ParametrizedDynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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

    def __init__(self, states, actions, model, next_states=None, distributions=None, preprocessors=None,
                 postprocessors=None):
        """
        Initialize the NN dynamic model.

        Args:
            states (State): state inputs.
            actions (Action): action inputs.
            next_states (State, None): state outputs. If None, it will take the state inputs as the outputs.
            model (NNApproximator, NN): neural network model.
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if model is None:
            raise TypeError("Expecting the model to be a neural network and not None.")
        elif not isinstance(model, NNApproximator):
            if next_states is None:
                next_states = states
            model = NNApproximator(inputs=[states, actions], outputs=next_states, model=model,
                                   preprocessors=preprocessors, postprocessors=postprocessors)
        super(NNDynamicModel, self).__init__(states, actions, model=model, next_states=next_states,
                                             distributions=distributions)


class MLPDynamicModel(NNDynamicModel):
    r"""MLP Dynamic Model

    """

    def __init__(self, states, actions, next_states=None, hidden_units=(), activation_fct='Linear',
                 last_activation_fct=None, dropout_prob=None, distributions=None, preprocessors=None,
                 postprocessors=None):
        """
        Initialize the multi-layer perceptron model.

        Args:
            states (State): state inputs.
            actions (Action): action inputs.
            next_states (State, None): state outputs. If None, it will take the state inputs as the outputs.
            hidden_units (tuple, list of int): number of hidden units in each layer
            activation_fct (str): activation function to apply on each layer
            last_activation_fct (str, None): activation function to apply on the last layer
            dropout_prob (None, float): dropout probability
            distributions (torch.distributions.Distribution): distribution to use to sample the next state. If None,
                it will be deterministic.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
        """
        if next_states is None:
            next_states = states
        model = MLPApproximator(inputs=[states, actions], outputs=next_states, hidden_units=hidden_units,
                                activation_fct=activation_fct, last_activation_fct=last_activation_fct,
                                dropout_prob=dropout_prob)
        super(MLPDynamicModel, self).__init__(states, actions, model=model, next_states=next_states,
                                              distributions=distributions, preprocessors=preprocessors,
                                              postprocessors=postprocessors)
