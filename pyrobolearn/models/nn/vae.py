#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Variational Auto-Encoder (VAE) learning model.

This file provides the VAE model; a parametric, generally non-linear, non-recurrent, generative, and stochastic model.
This model is a generative latent variable model which projects the input data into a latent lower
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

from pyrobolearn.distributions.modules import MeanModule, DiagonalCovarianceModule, GaussianModule
from pyrobolearn.models.nn.ae import AE


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class VAE(AE):
    r"""Variational AutoEncoder

    Type: generative model

    Variational Auto-encoders (VAEs) are generative latent models that projects the input data to a latent space,
    where a probability distribution is defined over it, and latent vector which are sampled are projected back to
    the input space. This allows later to generate

    The loss being minimized with VAEs is the reconstruction/generation loss and the latent loss which measures how
    far the latent distribution produced by the encoder is from a predefined distribution. This predefined latent
    distribution is often selected to be a unit Gaussian.

    .. seealso:: Generative Adversarial Networks

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
        [2] "Tutorial on Variational Autoencoders", Doersch, 2016
        [3] "Variational Autoencoders Explained" (http://kvfrans.com/variational-autoencoders-explained/), Frans, 2016
    """

    def __init__(self, encoder, decoder, latent_distribution, predefined_latent_distribution, input_shape,
                 output_shape):
        """
        Initialize the Variational Auto-Encoder.

        Args:
            encoder (torch.nn.Module): encoder module.
            decoder (torch.nn.Module): decoder module.
            latent_distribution (torch.nn.Module): latent distribution module. Given the encoder's output,
                the :attr:`latent_distribution` module outputs a `torch.distribution.Distribution`.
                See `pyrobolearn/distributions/modules.py` for more choices.
            predefined_latent_distribution (torch.distributions.Distribution): Predefined latent distribution to which
            the predicted latent distribution should be close to.
            input_shape (tuple of int): input shape.
            output_shape (tuple of int): output shape.
        """
        super(VAE, self).__init__(encoder, decoder, input_shape, output_shape)

        # check latent distribution module
        if not isinstance(latent_distribution, torch.nn.Module):
            raise TypeError("Expecting the given latent distribution to be an instance of `torch.nn.Module`, instead "
                            "got: {}".format(latent_distribution))

        # check predefined latent distribution
        if not isinstance(predefined_latent_distribution, torch.distributions.Distribution):
            raise TypeError("Expecting the predefined latent distribution to be an instance of "
                            "`torch.distributions.Distribution`, instead got: "
                            "{}".format(predefined_latent_distribution))

        # check that the distribution returned by the latent distribution module is the same as the predefined latent
        # distribution
        x = torch.rand(input_shape).unsqueeze(0)
        x = self.encode(x)
        distribution = latent_distribution(x)
        if not isinstance(distribution, predefined_latent_distribution.__class__):
            raise ValueError("The distribution returned by the latent distribution module (i.e. {}) is not an instance "
                             "of the predefined latent distribution "
                             "(i.e. {})".format(type(distribution), type(predefined_latent_distribution)))

        self.latent_distribution = latent_distribution
        self.predefined_distribution = predefined_latent_distribution

    ##################
    # Static methods #
    ##################

    @staticmethod
    def is_discriminative():
        """A neural network is a discriminative model which given inputs predicts some outputs"""
        return True

    @staticmethod
    def is_generative():  # unless VAE, GAN,...
        """VAEs are generative probabilistic models that learn a latent space."""
        return True

    ###########
    # Methods #
    ###########

    def forward(self, x):
        """Forward the input to the encoder and decoder."""
        # go through the encoder, sample from the latent distribution, and decode the samples
        shape = (len(x),) if len(x.shape) > 1 else (1,)
        x = self.encoder(x)
        x = self.latent_distribution(x).rsample(sample_shape=shape).squeeze(0)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """Run the encoder."""
        return self.encoder(x)

    def decode(self, x):
        """Run the decoder."""
        return self.decoder(x)

    def sample_latent(self, shape=(), seed=None):
        """Sample latent vectors."""
        if seed is not None:
            torch.manual_seed(seed)
        return self.predefined_distribution.rsample(sample_shape=shape)

    def sample(self, shape=(), seed=None):
        """Sample from the VAE."""
        if seed is not None:
            torch.manual_seed(seed)
        x = self.predefined_distribution.rsample(sample_shape=shape)
        return self.decoder(x)


class MLP_VAE(VAE):
    r"""Multi-Layer Perceptron Variational Auto-Encoder
    """

    def __init__(self, encoder_units=[], decoder_units=None, activation=None, last_activation=None,
                 dropout=None, latent_distribution=None, predefined_latent_distribution=None):
        """
        Initialize the MLP VAE.

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
            latent_distribution (torch.nn.Module, None): latent distribution module. Given the encoder's output,
                the :attr:`latent_distribution` module outputs a `torch.distribution.Distribution`. If None, it will
                create a Gaussian module which outputs a Gaussian distribution based on a learned mean and diagonal
                covariance. See `pyrobolearn/distributions/modules.py` for more choices.
            predefined_latent_distribution (torch.distributions.Distribution, None): Predefined latent distribution to
                which the predicted latent distribution should be close to.
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

        # latent distribution module
        if latent_distribution is None:
            size = encoder_units[-1]
            mean = MeanModule(num_inputs=size, num_outputs=size)
            covariance = DiagonalCovarianceModule(num_inputs=size, num_outputs=size)
            latent_distribution = GaussianModule(mean=mean, covariance=covariance)

        # predefined latent distribution
        if predefined_latent_distribution is None:
            size = encoder_units[-1]
            mean = torch.zeros(size)
            covariance = torch.diag(torch.ones(size))
            predefined_latent_distribution = torch.distributions.MultivariateNormal(loc=mean,
                                                                                    covariance_matrix=covariance)

        super(MLP_VAE, self).__init__(encoder, decoder, latent_distribution=latent_distribution,
                                      predefined_latent_distribution=predefined_latent_distribution,
                                      input_shape=tuple([num_units[0]]),
                                      output_shape=tuple([num_units[-1]]))


# Tests
if __name__ == '__main__':
    # create MLP network
    vae = MLP_VAE(encoder_units=(4, 3, 2), activation='relu')
    print(vae)

    x = torch.rand(4).unsqueeze(0)
    y = vae.forward(x)
    print("Input: {} - Output: {}".format(x, y))
