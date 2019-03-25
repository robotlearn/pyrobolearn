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
__license__ = "MIT"
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
        model = NN(model, input_size=..., output_size=...)

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
        [2] PyTorch: https://pytorch.org/
        nn.Module: https://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module
# - nn.Sequential: https://pytorch.org/docs/master/_modules/torch/nn/modules/container.html#Sequential
    """

    def __init__(self, model, input_shape, output_shape, framework=None):
        r"""Initialize the NN model.

        Args:
            model (torch.nn.Module or keras.base_layer.Layer): pytorch/keras model
            input_shape (int, tuple/list of int): dimensions of the input
            output_shape (int, tuple/list of int): dimensions of the output
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
        self._input_shape = input_shape
        self._output_shape = output_shape

        # TODO: infer the framework based on the model
        self.framework = framework

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the inner learning model."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the inner learning model."""
        if model is not None:
            if not (isinstance(model, torch.nn.Module)):  # or isinstance(model, keras.models.Model)):
                raise TypeError("The model should be an instance of torch.nn.Module or keras.models.Model")
        self._model = model

    @property
    def input_size(self):
        """Return the input size of the model."""
        return np.prod(self.input_shape)

    @property
    def output_size(self):
        """Return the output size of the model."""
        return np.prod(self.output_shape)

    @property
    def input_shape(self):  # TODO
        """Return the input shape."""
        return self._input_shape

    @property
    def output_shape(self):  # TODO
        """Return the output shape."""
        return self._output_shape

    @property
    def input_dim(self):
        """Return the input dimension."""
        return len(self.input_shape)

    @property
    def output_dim(self):
        """Return the output dimension."""
        return len(self.output_shape)

    @property
    def num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters."""
        return len(list(self.hyperparameters()))

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

    def parameters(self):
        """Return an iterator over the model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Return an iterator over the model parameters, yielding both the name and the parameter itself."""
        return self.model.named_parameters()

    def list_parameters(self):
        """Return a list of parameters."""
        return list(self.parameters())

    def hyperparameters(self):
        """Return an iterator over the model hyper-parameters; this includes the number of units per layer, the number
        of layers, the activation functions, etc."""
        raise NotImplementedError

    def named_hyperparameters(self):
        """Return an iterator over the model hyper-parameters, yielding both the name and the parameter itself."""
        raise NotImplementedError

    def list_hyperparameters(self):
        """Return a list of the hyper-parameters; this includes the number of units per layer, the number of layers,
        the activation functions, etc."""
        raise NotImplementedError

    def predict(self, x=None, to_numpy=False):
        """Predict the output given the input."""
        # convert to torch tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # predict output given input
        x = self.model(x)

        # return the output (and convert it to numpy if specified)
        if to_numpy:
            if x.requires_grad:
                return x.detach().numpy()
            return x.numpy()
        return x

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

    def __init__(self, model, input_shape, output_shape):
        super(NNTorch, self).__init__(model, input_shape, output_shape, framework='pytorch')

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
