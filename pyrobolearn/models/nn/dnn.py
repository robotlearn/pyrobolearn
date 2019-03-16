#!/usr/bin/env python
"""Define the Deep Neural Network (DNN) learning model.

This file provides the DNN model; a parametric, generally non-linear, possibly recurrent, discriminative/generative,
and deterministic/probabilistic model.

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

# from pyrobolearn.models import Model

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NN(object):  # Model
    r"""Neural Network class

    This class describes the neural network model. It is basically a wrapper around deep learning frameworks such
    as pytorch, Keras, tensorflow and others. This class is inherited by any other neural network classes, such as
    convolution neural networks, recurrent neural networks, and so on.

    Note that we currently mainly focus on `PyTorch`.

    * PyTorch (https://pytorch.org/)
        * PyTorch is ... PyTorch allows for dynamic ...
        * torch.nn.Module: it represents the base class for all the neural networks / layers
        * torch.nn.modules.loss(torch.nn.Module): it contains the definition of some popular losses
        * torch.optim.Optimizer: it is the base class for all the optimizers

    Examples::

        import torch.nn as nn

        model = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3), Flatten(), nn.Linear(320, 10))
        model = NN(model, input_dims=..., output_dims=...)

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
        [2] PyTorch: https://pytorch.org/
        nn.Module: https://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module
# - nn.Sequential: https://pytorch.org/docs/master/_modules/torch/nn/modules/container.html#Sequential
    """

    def __init__(self, model, input_dims, output_dims, framework=None):
        r"""Initialize the NN model.

        Args:
            model (torch.nn.Module or keras.base_layer.Layer): pytorch/keras model
            input_dims (int, tuple/list of int): dimensions of the input
            output_dims (int, tuple/list of int): dimensions of the output
        """
        super(NN, self).__init__()

        # check if given model is valid
        # if model is not None:
        #     if isinstance(model, torch.nn.Module):
        #         self.framework = 'pytorch'
        #     elif isinstance(model, keras.models.Model):
        #         self.framework = 'keras'
        #     else:
        #         raise TypeError("Model should be an instance of torch.nn.Module")

        # set model (written in the specified framework)
        self.model = model
        self.input_dims = input_dims
        self.output_dims = output_dims

        # TODO: infer the framework based on the model
        self.framework = framework

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            if not (isinstance(model, torch.nn.Module) or isinstance(model, keras.models.Model)):
                raise TypeError("The model should be an instance of torch.nn.Module or keras.models.Model")
        self._model = model

    @property
    def input_shape(self):  # TODO
        return self.input_dims

    @property
    def output_shape(self):  # TODO
        return self.output_dims

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_parametric():
        """A neural network is a parametric model"""
        return True

    @staticmethod
    def is_linear():  # unless all layers are linear
        """A neural network is in general non-linear, where non-linear activation functions are applied on each
         layer output. If all the activation layers are linear, then the NN is linear."""
        return False

    @staticmethod
    def is_recurrent():  # unless RNN
        """Unless the neural network is a RNN, it is not recurrent."""
        return False

    @staticmethod
    def is_probabilistic():  # unless variational methods (dropouts,...)
        """The neural network is not a probabilistic model per se. However, it can be simulated by using dropouts,
        or using a probabilistic distribution on the output of the last layer. For instance, the network can
        output the mean and covariance matrices which parametrizes a Gaussian probabilistic distribution"""
        return False  # if False then it is deterministic

    @staticmethod
    def is_discriminative():
        """A neural network is a discriminative model which given inputs predicts some outputs"""
        return True

    @staticmethod
    def is_generative():  # unless VAE, GAN,...
        """Standard neural networks are not generative, and thus we can not sample from it. This is different,
        for instance, for generative adversarial networks (GANs) and variational auto-encoders (VAEs)."""
        return False

    ###########
    # Methods #
    ###########

    def predict(self, x=None):
        return self.model(x)

    def get_input_dims(self):
        return self.input_dims

    def get_output_dims(self):
        return self.output_dims

    def parameters(self):
        return self.model.parameters()

    def get_params(self):
        return list(self.parameters())

    def get_hyperparams(self):
        """
        Return the number of units per layer, the number of layers, and the type of layers.
        """
        raise NotImplementedError

    def hyperparameters(self):
        raise NotImplementedError

    #############
    # Operators #
    #############

    def __str__(self):
        """
        Return string describing the NN model.
        """
        if self.framework == 'pytorch':
            return str(self.model)
        elif self.framework == 'keras':
            summary = []
            self.model.summary(print_fn=lambda s: summary.append(s))
            return '\n'.join(summary)
        else:
            raise NotImplementedError

    def __getitem__(self, key):
        """
        Return the corresponding layer(s).
        """
        return self.model[key]

    def __rshift__(self, other):
        """
        Concatenate two NN models in sequence, and return the sequenced model.
        Note that It doesn't modify the given models, but return a new one.

        Args:
            other (NN): other NN model

        Returns:
            NN: sequenced model
        """
        # copy current model
        model = copy.deepcopy(self.model)

        # concatenate the other model
        for idx, item in enumerate(other.model, start=len(self.model)):
            model.add_module(str(idx), item)

        # return the concatenation
        return NN(model)


class NNTorch(NN):
    r"""Neural Network written in PyTorch
    """

    def __init__(self, model, input_dims, output_dims):
        super(NNTorch, self).__init__(model, input_dims, output_dims, framework='pytorch')

    def save(self, filename):
        """
        Save the neural network to the specified file.

        Args:
            filename (str): filename to save the neural network
        """
        torch.save(self.model, filename)

    def load(self, filename):
        """
        Load the neural network from the specified file.

        Args:
            filename (str): filename from which to load the neural network
        """
        self.model = torch.load(filename)
        # check input and output dimensions

    def __str__(self):
        """
        Return string describing the NN model.
        """
        return str(self.model)
