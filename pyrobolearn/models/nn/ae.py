# -*- coding: utf-8 -*-
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

import torch

from pyrobolearn.models.nn.dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class AE(NN):
    r"""Auto-Encoder

    Auto-encoders are composed of an encoder and decoder parts. The encoder encodes/projects the input data into a
    lower dimensional space, while the decoder projects the latent data back to the output space. The output space is
    often the same as the input space.

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self, encoder, decoder, input_shape, output_shape):
        """
        Initialize the auto-encoder.

        Args:
            encoder (torch.nn.Module): encoder module.
            decoder (torch.nn.Module): decoder module.
            input_shape (tuple of int): input shape.
            output_shape (tuple of int): output shape.
        """
        self.encoder = encoder
        self.decoder = decoder
        # model = torch.nn.Sequential(*(list(encoder.modules())[1:] + list(decoder.modules())[1:]))
        model = torch.nn.Sequential(encoder, decoder)
        super(AE, self).__init__(model, input_shape, output_shape)

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_latent():
        """AEs are latent models."""
        return True

    ###########
    # Methods #
    ###########

    def forward(self, x):
        """Forward the input to the encoder and decoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """Run the encoder."""
        return self.encoder(x)

    def decode(self, x):
        """Run the decoder."""
        return self.decoder(x)


class MLP_AE(AE):
    r"""Multi-Layer Peceptron Auto-Encoder
    """

    def __init__(self, encoder_units=[], decoder_units=None, activation=None, last_activation=None, dropout=None):
        """
        Initialize the MLP AE.

        Args:
            encoder_units (list/tuple of int): number of units for each layer in the encoder (this includes the input
                layer)
            decoder_units (list/tuple of int, None): number of units for each layer in the decoder (this includes the
                output layer). If None, it will take the encoder units but in the reverse order.
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the number of
                                                                   hidden layers.
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
        """
        # check the encoder length
        if len(encoder_units) < 2:
            raise ValueError("Expecting more than the input layer for the encoder.")

        # check decoder units.
        if decoder_units is None:
            decoder_units = encoder_units[::-1]  # reverse
        else:
            decoder_units = encoder_units[-1:] + decoder_units
        if len(decoder_units) == 1:
            raise ValueError("Expecting at least the output layer for the decoder.")

        num_units = encoder_units + decoder_units[1:]

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
        def build_layers(units, encoder=True):
            layers = []
            size = len(units[:-1]) if encoder else len(units[:-2])
            for i in range(size):
                # add linear layer
                layer = torch.nn.Linear(units[i], units[i + 1])
                layers.append(layer)

                # add activation layer
                if activation is not None:
                    layers.append(activation())

                # add dropout layer
                if dropout_layer is not None:
                    layers.append(dropout_layer)

            # last output layer if decoder
            if not encoder:
                layers.append(torch.nn.Linear(units[-2], units[-1]))
                if last_activation is not None:
                    layers.append(last_activation)

            return layers

        encoding_layers = build_layers(encoder_units, encoder=True)
        decoding_layers = build_layers(decoder_units, encoder=False)

        # Encoder
        encoder = torch.nn.Sequential(*encoding_layers)
        # Decoder
        decoder = torch.nn.Sequential(*decoding_layers)

        super(MLP_AE, self).__init__(encoder, decoder, input_shape=tuple([num_units[0]]),
                                     output_shape=tuple([num_units[-1]]))


# Tests
if __name__ == '__main__':

    # create MLP network
    autoencoder = MLP_AE(encoder_units=(4, 3, 2), activation='relu')
    print(autoencoder)

    x = torch.rand(4)
    y = autoencoder(x)
    print("Input: {} - Output: {}".format(x, y))
