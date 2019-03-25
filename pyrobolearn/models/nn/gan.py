#!/usr/bin/env python
"""Define the Generative Adversarial Network (GAN) learning model.

This file provides the GAN model; a parametric, generally non-linear, non-recurrent, generative, and stochastic model.
This is a generative model which works in a game theory setting by having a generator and discriminator compete
between each other. The goal of the generator is to generate data samples that are similar to the provided dataset
and fool the discriminator. The goal of the discriminator is to discriminate the given samples by identifying the fake
ones from the true ones.

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

from pyrobolearn.models.nn.dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GAN(NN):
    r"""Generative Adversarial Network

    Type: generative model

    .. seealso:: Variational Auto-Encoders

    References:
        [1] "NIPS:
        [2]
    """

    def __init__(self):
        pass

    @staticmethod
    def isDiscriminative():
        """A neural network is a discriminative model which given inputs predicts some outputs"""
        return True

    @staticmethod
    def isGenerative():  # unless VAE, GAN,...
        """Standard neural networks are not generative, and thus we can not sample from it. This is different,
        for instance, for generative adversarial networks (GANs) and variational auto-encoders (VAEs)."""
        return True


class GANTorch(NNTorch):
    r"""Generative Adversarial Network in PyTorch
    """
    pass
