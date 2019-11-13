#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides value function approximators based on neural networks.

For instance, a value network is a function represented by a neural network that maps a state to a value (real number).
"""

from abc import ABCMeta
import torch

from pyrobolearn.models import NN
from pyrobolearn.approximators import NNApproximator, MLPApproximator
from pyrobolearn.values.value import ParametrizedValue, ParametrizedQValue, ParametrizedQValueOutput


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# class ValueNetwork(ParametrizedValue):
#     r"""Value Network
#
#     Use a neural network to approximate the value function.
#     """
#     __metaclass__ = ABCMeta
#
#     def __init__(self, state, model):
#         super(ValueNetwork, self).__init__(state)
#
#         # Check the given model
#         if not isinstance(model, NNApproximator):
#             if isinstance(model, NN):
#                 model = NNApproximator(state, torch.Tensor([1]), model)
#             else:
#                 if isinstance(model, torch.nn.Module):
#                     model = NN(model)
#                     model = NNApproximator(state, torch.Tensor([1]), model)
#                 else:
#                     raise TypeError("The model for the neural network is not an instance of model.NN or "
#                                     "torch.nn.Module")
#
#         self.model = model


class ValueNetwork(ParametrizedValue):
    r"""State Value Network

    This is defined by :math:`V_{\psi}(s_t)` where :math:`\psi` represents the network parameters.
    """

    def __init__(self, state, model):
        """
        Initialize the NN state value function approximator.

        Args:
            state (State): input state.
            model (NN, NNApproximator): Neural Network model / approximator.
        """
        super(ValueNetwork, self).__init__(state, model)


class QValueNetwork(ParametrizedQValue):
    r"""Q-Value Network (which accepts as inputs the states and actions)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a neural network, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and actions :math:`a`,
    and outputs the value :math:`Q(s,a)`. This can be used for continuous actions as well as discrete actions.
    """

    def __init__(self, state, action, model):
        """
        Initialize the NN state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input action.
            model (NN, NNApproximator): Neural Network model / approximator.
        """
        super(QValueNetwork, self).__init__(state, action, model)


class QValueOutputNetwork(ParametrizedQValueOutput):
    r"""Q-Value Output Network (which accepts as inputs the states and outputs a Q-value for each discrete action)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a neural network, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and outputs the value
    :math:`Q(s,a)` for each discrete action. This can NOT be used with continuous actions.
    """

    def __init__(self, state, action, model):
        """
        Initialize the NN state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output action.
            model (NN, NNApproximator): Neural Network model / approximator.
        """
        super(QValueOutputNetwork, self).__init__(state, action, model)


class MLPValue(ValueNetwork):
    r"""Multi-Layer Perceptron (MLP) State Value Function Approximator

    This is defined by :math:`V_{\psi}(s_t)` where the function :math:`V` is approximated by a multilayer perceptron.
    """

    def __init__(self, state, hidden_units=(), activation='linear', last_activation=None, dropout=None,
                 preprocessors=None):
        """Initialize the Value MLP approximator.

        Args:
            state (State): 1D-states that is feed to the policy (the input dimensions will be inferred from the
                            states)
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        output = torch.Tensor([1.])  # torch.Tensor([[1.]])
        model = MLPApproximator(state, output, hidden_units=hidden_units, activation=activation,
                                last_activation=last_activation, dropout=dropout, preprocessors=preprocessors)
        super(MLPValue, self).__init__(state, model)


class MLPQValue(QValueNetwork):
    r"""MLP Q-value function approximator (which accepts as inputs the states and actions)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a MLP model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and actions :math:`a`,
    and outputs the value :math:`Q(s,a)`. This can be used for continuous actions as well as discrete actions.
    """

    def __init__(self, state, action, hidden_units=(), activation='linear', last_activation=None, dropout=None,
                 preprocessors=None):
        """
        Initialize the MLP state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input action.
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = MLPApproximator(inputs=[state, action], outputs=torch.Tensor([1]), hidden_units=hidden_units,
                                activation=activation, last_activation=last_activation, dropout=dropout,
                                preprocessors=preprocessors)
        super(MLPQValue, self).__init__(state, action, model=model)


class MLPQValueOutput(ParametrizedQValueOutput):
    r"""MLP Q-value function approximator (which accepts as inputs the states and outputs a Q-value for each discrete
    action)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a MLP model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and outputs the value
    :math:`Q(s,a)` for each discrete action. This can NOT be used with continuous actions.
    """

    def __init__(self, state, action, hidden_units=(), activation='linear', last_activation=None, dropout=None,
                 preprocessors=None):
        """
        Initialize the MLP state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output action.
            hidden_units (list/tuple of int): number of hidden units in the corresponding layer
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = MLPApproximator(inputs=state, outputs=action, hidden_units=hidden_units, activation=activation,
                                last_activation=last_activation, dropout=dropout, preprocessors=preprocessors)
        super(MLPQValueOutput, self).__init__(state, action, model=model)
