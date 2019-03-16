#!/usr/bin/env python
"""Define the Auto-Encoder (AE) learning model.

This file provides the AE model; a parametric, generally non-linear, non-recurrent, discriminative,
and deterministic model. This model is a latent variable model which projects the input data into a latent lower
dimensional space through an encoder, and re-projects it to the original data space through the use of the decoder.

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

from dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AE(NN):
    r"""Auto-Encoder

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self):
        pass


class _AETorch(torch.nn.Module):
    r"""Auto-Encoder written in Pytorch (that inherits from `torch.nn.Module`)
    """

    def __init__(self, layer_sizes=[], activation_fct=None, dropout=None, encoder=None, decoder=None):
        super(_AETorch, self).__init__()

        if encoder is None and decoder is None:
            # nb of layers (the input layer doesn't count)
            self.num_layers = len(layer_sizes) - 1
            layers = [torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers)]

            # check activation function and insert it after each linear layer
            if activation_fct is not None:
                if isinstance(activation_fct, str):
                    activation_fct = getattr(torch.nn, activation_fct)()
                elif activation_fct.__module__ == 'torch.nn.modules.activation':
                    if inspect.isclass(activation_fct):
                        activation_fct = activation_fct()
                else:
                    raise ValueError("activation_fct should be a string or belong to torch.nn.modules.activation")

                # add activation layer
                for i in range(self.num_layers-1, 0, -1):
                    layers.insert(activation_fct)

            # check dropout
            if dropout is not None:
                if isinstance(dropout, float):
                    dropout = torch.nn.Dropout(dropout)
                elif dropout.__module__ == 'torch.nn.modules.dropout':
                    raise ValueError("Dropout should be a float or belong to torch.nn.modules.dropout")

                # add dropout layer
                for i in range(self.num_layers-1, 0, -2):
                    layers.insert(dropout)

            # Encoder
            encoder = torch.nn.Sequential(*layers[:len(layers)//2])
            # Decoder
            decoder = torch.nn.Sequential(*layers[len(layers)//2:])

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class AETorch(NNTorch):
    r"""Auto-Encoder in Pytorch
    """
    pass
