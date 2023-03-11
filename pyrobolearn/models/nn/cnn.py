#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Convolutional Neural Network (CNN) learning model.

This file provides the CNN model; a parametric, generally non-linear, non-recurrent, discriminative,
and deterministic model. This model is convenient for data arrays/tensors that have cells that have a spatial
relationship between them. For instance, pictures are 2D or 3D arrays where each pixel is related with its neighbors.

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
import math

from pyrobolearn.models.nn.dnn import NN


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Flatten(torch.nn.Module):
    r"""Flatten layer."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(NN):
    r"""(Feed-forward) Convolutional Neural Networks

    Convolutional neural networks are neural networks that captures the spatial relationship between data features.
    For instance, they are used for pictures where neighboring pixels are related to each other.

    References:
        [1] "CS231n: Convolutional Neural Networks for Visual Recognition", Li et al., 2019
    """

    def __init__(self, model, input_shape, output_shape):
        super(CNN, self).__init__(model, input_shape, output_shape)

    def forward(self, x):
        """Forward the inputs in the network."""
        # reshape the inputs if necessary
        unsqueezed = True if len(x.shape) == 3 else False
        if unsqueezed:
            x = x.unsqueeze(0)

        # feed the inputs to the model
        x = self.model(x)

        # reshape the outputs
        if unsqueezed:
            x = x.squeeze(0)

        # return the output
        return x


class CNN2D(CNN):
    r"""(Feed-forward) Convolutional Neural Networks

    Convolutional neural networks are neural networks that captures the spatial relationship between data features.
    For instance, they are used for pictures where neighboring pixels are related to each other.

    References:
        [1] "CS231n: Convolutional Neural Networks for Visual Recognition", Li et al., 2019
    """

    def __init__(self, units=[], activation='ReLU', last_activation=None, pool=None, dropout=None, dropout2d=None,
                 use_batch_norm=False, model=None):
        """
        Initialize the convolutional neural network.

        Args:
            units (list/tuple of int): number of units per layer. For instance,
                `units=[(3,32,32), (3,6,5), [2,2], 10, 2]` means the input shape is (3,32,32), the next layer
                is a convolutional layer with `in_channels=3`, `out_channels=6`, `kernel_size=5`, the next layer is
                a pooling layer with a kernel_size of 2 and a stride of 2, the next layer is a flatten layer which
                feeds the output to a linear layer of 10 units to finally finish with 2 units in the output layer.
                Use tuples to specify convolutional layers and lists to specify pooling layers.
            activation (str, torch.nn.Module, None): activation function to be used after convolution layers and linear
                layers.
            last_activation (str, torch.nn.Module, None): last activation function to be used at the output layer.
            pool (torch.nn.Module, None): pooling layer.
            dropout2d (float, None): dropout probability for 2d layers. If None, the probability is 0. This value
                should be smaller than the dropout probability for 1d layer (i.e. below at least 0.2).
            dropout (float, None): dropout probability for 1d layers. If None, the probability is 0.
            use_batch_norm (bool): If True, it will use batch norm.
            model (torch.nn.Module, None): If the model is given, it will not use the previous defined
        """
        # create convolutional neural network if necessary
        if model is None:
            # check number of units
            if len(units) < 2:
                raise ValueError("The num_units list/tuple needs to have at least the input and output layers")

            # check that the input shape is 2d
            if len(units[0]) != 2 and len(units[0]) != 3:
                raise ValueError("Expecting the input shape to be (Height, Width) or (Channel, Height, Width), "
                                 "instead got a length of: {}".format(len(units[0])))

            # check for activation fcts
            activations = dir(torch.nn.modules.activation)
            activations = {act: act for act in activations}
            activations.update({act.lower(): act for act in activations})

            def check_activation(activation):
                if activation is None or activation.lower() == 'linear':
                    activation = None
                else:
                    if activation not in activations:
                        raise ValueError("The given activation function {} is not available".format(activation))
                    activation = getattr(torch.nn, activations[activation])
                return activation

            activation = check_activation(activation)
            last_activation = check_activation(last_activation)

            # check dropout
            dropout_layer, dropout2d_layer = None, None
            if dropout is not None:
                dropout_layer = torch.nn.Dropout(dropout)
            if dropout2d is not None:
                dropout2d_layer = torch.nn.Dropout2d(dropout2d)

            # check pooling layer
            pooling_layer = None
            if pool is not None:
                if isinstance(pool, torch.nn.Module):
                    pooling_layer = pool
                elif isinstance(pool, str):
                    pools = {item: item for item in dir(torch.nn) if item[-6:] == 'Pool2d'}
                    pools.update({item.lower(): item for item in pools})
                    if pool not in pools:
                        raise ValueError("The given pooling layer {} has not been implemented".format(pool))
                    pooling_layer = getattr(torch.nn, pools[pool])
                else:
                    pass

            # keep track of (C, H, W) dimensions
            if len(units[0]) == 2:
                channel = 1
                height, width = units[0]
            else:  # elif len(units[0]):
                channel, height, width = units[0]

            # build network
            layers = []
            linear_layer = False
            for i in range(1, len(units)):

                # check if last layer
                if i == len(units) - 1:
                    if isinstance(units[i], int):

                        # if first linear layer, add flatten layer
                        in_features = units[i-1]
                        if i - 1 > 0 and not isinstance(units[i-1], int):
                            layers.append(Flatten())
                            in_features = channel * height * width

                        # add linear layer
                        layers.append(torch.nn.Linear(in_features, units[i]))

                        # get out of the loop
                        break

                # convolution layer
                if isinstance(units[i], tuple):
                    if linear_layer:
                        raise ValueError("Got a convolution layer after a linear layer... This is not supported.")

                    # add convolution layer
                    unit = units[i]
                    if len(units[i]) == 2:
                        unit = (channel,) + units[i]
                    layer = torch.nn.Conv2d(*unit)
                    layers.append(layer)

                    # compute new dimensions
                    channel = layer.out_channels
                    p, d, k, s = layer.padding, layer.dilation, layer.kernel_size, layer.stride
                    height = int(math.floor((height + 2. * p[0] - d[0] * (k[0] - 1) - 1.) / s[0] + 1))
                    width = int(math.floor((width + 2. * p[1] - d[1] * (k[1] - 1) - 1.) / s[1] + 1))

                    # if use batch normalization
                    if use_batch_norm:
                        layers.append(torch.nn.BatchNorm2d(num_features=unit[1]))

                    # add activation layer
                    if activation is not None:
                        layers.append(activation())

                    # add dropout layer if specified
                    if dropout2d_layer is not None:
                        layers.append(dropout2d_layer)

                # pooling layer
                elif isinstance(units[i], list):
                    if linear_layer:
                        raise ValueError("Got a pooling layer after a linear layer... This is not supported.")

                    if pooling_layer is not None:
                        layer = pooling_layer(*units[i])
                        layers.append(layer)

                        # compute new dimensions
                        p, d, k, s = layer.padding, layer.dilation, layer.kernel_size, layer.stride
                        if isinstance(p, int):
                            p = (p, p)
                        if isinstance(k, int):
                            k = (k, k)
                        if isinstance(s, int):
                            s = (s, s)
                        if isinstance(d, int):
                            d = (d, d)
                        height = int(math.floor((height + 2. * p[0] - d[0] * (k[0] - 1) - 1.) / s[0] + 1))
                        width = int(math.floor((width + 2. * p[1] - d[1] * (k[1] - 1) - 1.) / s[1] + 1))

                # linear layer
                elif isinstance(units[i], int):
                    linear_layer = True
                    # if first linear layer, add flatten layer
                    in_features = units[i-1]
                    if not isinstance(units[i-1], int):  # before last layer was convolutional layer
                        layers.append(Flatten())
                        in_features = channel * height * width

                    # add linear layer
                    layers.append(torch.nn.Linear(in_features, units[i]))

                    # add batch normalization layer
                    if use_batch_norm:
                        layers.append(torch.nn.BatchNorm1d(num_features=units[i + 1]))

                    # add activation layer
                    if activation is not None:
                        layers.append(activation())

                    # add dropout layer if specified
                    if dropout_layer is not None:
                        layers.append(dropout_layer)

                else:
                    raise TypeError("One of the units is not an int, tuple, or list, instead got: "
                                    "{}".format(type(units[i])))

            # add last activation function
            if last_activation is not None:
                layers.append(last_activation)

            # create nn model
            model = torch.nn.Sequential(*layers)

        # define input and output shapes
        input_shape = tuple([units[0]]) if isinstance(units[0], int) else units[0]
        output_shape = tuple([units[-1]]) if isinstance(units[-1], int) else units[-1]

        super(CNN2D, self).__init__(model, input_shape=input_shape, output_shape=output_shape)


# Tests
if __name__ == '__main__':
    input_shape = (3, 32, 32)

    # create convolutional neural network
    cnn = CNN2D(units=[input_shape, (3, 6, 5), [2, 2], (6, 16, 5), [2, 2], 120, 84, 10], activation='relu',
                pool='MaxPool2d')
    print(cnn)

    x = torch.rand(*input_shape)
    y = cnn.forward(x)
    print("Input: {} - Output: {}".format(x, y))
