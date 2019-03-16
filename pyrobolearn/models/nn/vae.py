#!/usr/bin/env python
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


class VAE(NN):
    r"""Variational AutoEncoder

    Type: generative model

    .. seealso:: Generative Adversarial Networks

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
        [2] "Tutorial on Variational Autoencoder"
    """
    def __init__(self, layer_sizes, activation_fct=None, dropout=None):
        self.encoder = None
        self.decoder = None

    @staticmethod
    def isDiscriminative():
        """A neural network is a discriminative model which given inputs predicts some outputs"""
        return True

    @staticmethod
    def isGenerative():  # unless VAE, GAN,...
        """Standard neural networks are not generative, and thus we can not sample from it. This is different,
        for instance, for generative adversarial networks (GANs) and variational auto-encoders (VAEs)."""
        return True

    ###########
    # Methods #
    ###########

    def sample(self, size=None, seed=None):
        """Sample from the VAE"""
        pass


class VAETorch(NNTorch):
    r"""Variational AutoEncoder in PyTorch
    """
    pass
