#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the Linear Model.

The linear model is a discriminative deterministic model given by: :math:`y = f(x) = w^T x`.
"""

import copy
import types
import collections.abc

try:
    import cPickle as pickle
except ImportError as e:
    import pickle

import numpy as np
import torch

# from pyrobolearn.models.model import Model

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Linear(object):
    r"""Linear Model

    This class describes the linear parametric model: :math:`y = W x + b` where :math:`x` and :math:`y` are
    respectively the input and output vectors, :math:`W` is the weight matrix, and :math:`b` is the bias/intercept.
    """

    def __init__(self, num_inputs, num_outputs, add_bias=True):
        r"""
        Initialize the affine/linear model described mathematically by:

        .. math:: y = W x + b

        Args:
            num_inputs (int): dimension of the input
            num_outputs (int):  dimension of the output
            add_bias (bool): if True, it will add a bias to the prediction
        """
        super(Linear, self).__init__()
        self.model = torch.nn.Linear(num_inputs, num_outputs, bias=add_bias)
        self._num_parameters = len(self.get_vectorized_parameters())

    ##############
    # Properties #
    ##############

    @property
    def input_size(self):
        """Return the input size of the model"""
        return self.model.weight.shape[1]

    @property
    def output_size(self):
        """Return the output size of the model"""
        return self.model.weight.shape[0]

    @property
    def input_shape(self):
        """Return the input shape of the model"""
        return tuple([self.input_size])

    @property
    def output_shape(self):
        """Return the output shape of the model"""
        return tuple([self.output_size])

    @property
    def input_dim(self):
        """Return the input dimension; i.e. len(input_shape)."""
        return len(self.input_shape)

    @property
    def output_dim(self):
        """Return the output dimension; i.e. len(output_shape)."""
        return len(self.output_shape)

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self._num_parameters

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other, deep=True):
        """Return another copy of the linear model"""
        if not isinstance(other, Linear):
            raise TypeError("Trying to copy an object which is not a Linear model")
        if deep:
            return copy.deepcopy(other)
        return copy.copy(other)

    @staticmethod
    def is_parametric():
        """The linear model is a parametric model"""
        return True

    @staticmethod
    def is_linear():
        """By definition, a linear model is linear"""
        return True

    @staticmethod
    def is_recurrent():
        """The linear model is not recurrent; current outputs do not depend on previous inputs"""
        return False

    @staticmethod
    def is_deterministic():
        """The linear model is a deterministic model"""
        return True

    @staticmethod
    def is_probabilistic():
        """The linear model is not a probabilistic model; it is a deterministic one"""
        return False

    @staticmethod
    def is_discriminative():
        """The linear model is a discriminative model which predicts :math:`y = Wx + b` where :math:`x` is the input,
        and :math:`y` is the output"""
        return True

    @staticmethod
    def is_generative():
        """The linear model is not a generative model, and thus we can not sample from it"""
        return False

    @staticmethod
    def load(filename):
        """
        Load a model from memory.

        Args:
            filename (str): file that contains the model.
        """
        return pickle.load(filename)

    ###########
    # Methods #
    ###########

    def train(self):
        """Set into training mode."""
        self.model.train()

    def eval(self):
        """Set into eval mode."""
        self.model.eval()

    def copy_parameters(self, parameters):
        """Copy the given parameters.

        Args:
            parameters (Linear, torch.nn.Module, generator, iterable): the other model's parameters to copy.
        """
        if isinstance(parameters, self.__class__):
            self.model.load_state_dict(parameters.model.state_dict())
        elif isinstance(parameters, torch.nn.Module):
            self.model.load_state_dict(parameters.state_dict())
        elif isinstance(parameters, (types.GeneratorType, collections.abc.Iterable)):
            for model_params, other_params in zip(self.parameters(), parameters):
                model_params.data.copy_(other_params.data)
        else:
            raise TypeError("Expecting the given parameters to be an instance of `Linear`, `torch.nn.Module`, "
                            "`generator`, or an iterable object, instead got: {}".format(type(parameters)))

    def parameters(self):
        """Returns an iterator over the model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        return self.model.named_parameters()

    def list_parameters(self):
        """Return a list of parameters"""
        return list(self.parameters())

    def get_vectorized_parameters(self, to_numpy=True):
        """Return a vectorized form (1 dimensional array) of the parameters."""
        parameters = self.parameters()
        vector = torch.cat([parameter.reshape(-1) for parameter in parameters])  # np.concatenate = torch.cat
        if to_numpy:
            return vector.detach().numpy()
        return vector

    def set_vectorized_parameters(self, vector):
        """Set the vector parameters."""
        # convert the vector to torch array
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()

        # set the parameters from the vectorized one
        idx = 0
        for parameter in self.parameters():
            size = parameter.nelement()
            parameter.data = vector[idx:idx+size].reshape(parameter.shape)
            idx += size

    def predict(self, x, to_numpy=True):
        r"""
        Predict output vector :math:`y` given input vector :math:`x`, using the formula: :math:`y = W x + b`

        Args:
            x (np.ndarray, torch.Tensor): input vector
            to_numpy (bool): if True, return a np.array

        Returns:
            np.ndarray, torch.Tensor: output vector
        """
        # convert from numpy to pytorch if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # predict
        y = self.model(x)

        # return the output and convert it if necessary
        if to_numpy:
            if y.requires_grad:
                return y.detach().numpy()
            return y.numpy()
        return y

    def fit(self, X, Y):
        r"""Train the linear model using Linear Regression.

        This is performed by minimizing the L2 loss with respect to the parameters:

        .. math:: \min_W || Y - XW ||^2

        where :math:`X \in \mathcal{R}^{N \times (D_x+1)}` is the augmented input data matrix, and
        :math:`Y \in \mathcal{R}^{N \times (D_y)}` is the output data matrix.

        The best set of weights can be obtained by solving in closed-loop the above optimization process.
        The optimal solution is given by the pseudo-inverse: :math:`W^* = (X^\top X)^{-1} X^T Y`.

        Args:
            X (np.array[N,Dx], torch.Tensor[N,Dx]): input data matrix.
            Y (np.array[N,Dy], torch.Tensor[N,Dy]): output data matrix.
        """
        pass

    def reset(self):
        """Reset the linear model."""
        pass

    def __call__(self, x, to_numpy=True):
        """Predict the output given the input :attr:`x`."""
        return self.predict(x, to_numpy=to_numpy)

    # def concatenate(self, other):
    #     """
    #     Concatenate a linear model with another one.
    #
    #     Args:
    #         other (Linear): the other linear model
    #     """
    #     if not isinstance(other, Linear):
    #         raise TypeError("Expecting the other model to be also linear")
    #     # self.model.add_module()


# Tests
if __name__ == '__main__':
    # test with numpy
    x = np.array(range(3))
    model = Linear(num_inputs=len(x), num_outputs=2, add_bias=True)
    print("Linear model's input size: {}".format(model.input_size))
    print("Linear model's output size: {}".format(model.output_size))
    y = model(x)
    print("Linear input: {}".format(x))
    print("Linear output: {}".format(y))

    # test with pytorch
    x = torch.from_numpy(x).float()
    y = model(x, to_numpy=False)
    print("Linear input: {}".format(x))
    print("Linear torch output: {}".format(y))
    y = model(x, to_numpy=True)
    print("Linear numpy output: {}".format(y))
