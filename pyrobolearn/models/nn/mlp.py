#!/usr/bin/env python
"""Define the Multi-Layer Perceptron (MLP) learning model.

This file provides the MLP model; a parametric, generally non-linear, non-recurrent, discriminative,
and deterministic model.

We decided to use the `pytorch` framework because of its popularity in the research community field, flexibility,
similarity with numpy (but with automatic differentiation: autograd), GPU capabilities, and more Pythonic approach.
While we thought about using other frameworks (such as Keras, Tensorflow, and others) as well, it would have
unnecessarily complexify the whole framework, as these frameworks would not only influence the learning models,
but also the losses, optimizers, and other modules. While we could have written some interfaces that makes the bridge
between these various frameworks and ours, we came to the conclusion that this would take a considerable amount of
efforts and time that we do not have for the moment.

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
"""

import copy
import inspect
import numpy as np
import torch

from dnn import NN, NNTorch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MLP(NN):
    r"""Multi-Layer Perceptron

    Feed-forward and fully-connected neural network, where linear layers are followed by non-linear activation
    functions.

    .. math::

        h_{l} = f_{l}(W_{l} h_{l-1} + b_{l})

    where :math:`l \in [1,...,L]` with :math:`L` is the total number of layers,
    :math:`W_{l}` and :math:`b_{l}` are the weight matrix and bias vector at layer :math:`l`, :math:`f_{l}` is
    the nonlinear activation function, and :math:`h_{0} = x` and :math:`y = h_{L}` are the input and output vectors.

    The parameters of the neural networks are all the weight matrices and bias vectors.
    """

    def __init__(self, num_units=(), activation_fct='Linear', last_activation_fct=None, dropout_prob=None,
                 framework='pytorch'):
        """
        Initialize a MLP network.

        Args:
            num_units (list/tuple of int): number of units in each layer (this includes the input and output layer)
            activation_fct (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation_fct (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout_prob (None, float, or list/tuple of float/None): dropout probability.
            framework (str): specifies which framework we want to use between 'pytorch' and 'keras' (default: 'pytorch')
        """

        # check framework
        framework = framework.lower()
        if framework == 'pytorch':
            model = MLPTorch(num_units, activation_fct, last_activation_fct, dropout_prob)
        elif framework == 'keras':
            model = MLPKeras(num_units, activation_fct, last_activation_fct, dropout_prob)
        else:
            raise ValueError("The current frameworks allowed are pytorch and keras")

        # instantiate the super class
        super(MLP, self).__init__(model.model, input_dims=num_units[0], output_dims=num_units[-1], framework=framework)

        # rewrite methods
        self.save = model.save
        self.load = model.load
        self.__str__ = model.__str__


class MLPTorch(NNTorch):
    r"""Multi-Layer Perceptron in PyTorch

    Feed-forward and fully-connected neural network, where linear layers are followed by non-linear activation
    functions.

    .. math::

        h_{l} = f_{l}(W_{l} h_{l-1} + b_{l})

    where :math:`l \in [1,...,L]` with :math:`L` is the total number of layers,
    :math:`W_{l}` and :math:`b_{l}` are the weight matrix and bias vector at layer :math:`l`, :math:`f_{l}` is
    the nonlinear activation function, and :math:`h_{0} = x` and :math:`y = h_{L}` are the input and output vectors.

    The parameters of the neural networks are all the weight matrices and bias vectors.
    """

    def __init__(self, num_units=(), activation_fct='Linear', last_activation_fct=None, dropout_prob=None):
        """
        Initialize a MLP network.

        Args:
            num_units (list/tuple of int): number of units in each layer (this includes the input and output layer)
            activation_fct (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the
            last_activation_fct (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout_prob (None, float, or list/tuple of float/None): dropout probability.
        """
        # check number of units
        if len(num_units) < 2:
            raise ValueError("The num_units list/tuple needs to have at least the input and output layers")

        # set the dimensions of the input and output
        self.input_dims = num_units[0]
        self.output_dims = num_units[-1]

        # check for activation fcts
        activations = dir(torch.nn.modules.activation)
        activations = {act: act for act in activations}
        activations.update({act.lower(): act for act in activations})

        def check_activation(activation):
            if activation is None or activation.lower() == 'linear':
                activation = None
            else:
                if activation not in activations:
                    raise ValueError("The given activation function is not available")
                activation = getattr(torch.nn, activations[activation])
            return activation

        activation_fct = check_activation(activation_fct)
        last_activation_fct = check_activation(last_activation_fct)

        # check dropout
        dropout_layer = None
        if dropout_prob is not None:
            dropout_layer = torch.nn.Dropout(dropout_prob)

        # build pytorch network
        layers = []
        for i in range(len(num_units[:-2])):
            # add linear layer
            layer = torch.nn.Linear(num_units[i], num_units[i + 1])
            layers.append(layer)

            # add activation layer
            if activation_fct is not None:
                layers.append(activation_fct())

            # add dropout layer
            if dropout_layer is not None:
                layers.append(dropout_layer)

        # last output layer
        layers.append(torch.nn.Linear(num_units[-2], num_units[-1]))
        if last_activation_fct is not None:
            layers.append(last_activation_fct)

        # create nn model
        model = torch.nn.Sequential(*layers)

        super(MLPTorch, self).__init__(model, input_dims=num_units[0], output_dims=num_units[-1])


# Tests
if __name__ == '__main__':

    # create MLP network
    mlp = MLPTorch(num_units=(2,10,3), activation_fct='relu')
    print(mlp)
