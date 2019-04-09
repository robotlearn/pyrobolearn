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

import torch

from pyrobolearn.models.nn.dnn import NN

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

    def __init__(self, units=(), activation=None, last_activation=None, dropout=None):
        """
        Initialize a MLP network.

        Args:
            units (list/tuple of int): number of units in each layer (this includes the input and output layer)
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the number of
                                                                   hidden layers. If None, it is a linear layer.
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
        """
        # check number of units
        if len(units) < 2:
            raise ValueError("The num_units list/tuple needs to have at least the input and output layers")

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

        activation = check_activation(activation)
        last_activation = check_activation(last_activation)

        # check dropout
        dropout_layer = None
        if dropout is not None:
            dropout_layer = torch.nn.Dropout(dropout)

        # build pytorch network
        layers = []
        for i in range(len(units[:-2])):
            # add linear layer
            layer = torch.nn.Linear(units[i], units[i + 1])
            layers.append(layer)

            # add activation layer
            if activation is not None:
                layers.append(activation())

            # add dropout layer
            if dropout_layer is not None:
                layers.append(dropout_layer)

        # last output layer
        layers.append(torch.nn.Linear(units[-2], units[-1]))
        if last_activation is not None:
            layers.append(last_activation)

        # create nn model
        model = torch.nn.Sequential(*layers)

        super(MLP, self).__init__(model, input_shape=tuple([units[0]]), output_shape=tuple([units[-1]]))

        # rewrite methods
        # self.save = model.save
        # self.load = model.load
        # self.__str__ = model.__str__


# Tests
if __name__ == '__main__':

    # create MLP network
    mlp = MLP(units=(2, 10, 3), activation='relu')
    print(mlp)

    x = torch.rand(2)
    y = mlp(x)
    print("Input: {} - Output: {}".format(x, y))
