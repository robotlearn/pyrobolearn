#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Define the polynomial learning model.

The polynomial model is a discriminative deterministic model given by: :math:`y = f(x) = W \phi(x)`, where
:math:`\phi` is a function that returns a transformed input vector (possibly of higher dimension).
"""

import copy
# import inspect
import collections.abc
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


class PolynomialFunction(object):
    r"""Polynomial function

    Polynomial function to be applied on the given inputs.
    """

    def __init__(self, degree=1):
        """
        Initialize the polynomial function.

        Args:
            degree (int, list of ints): degree(s) of the polynomial. Setting `degree=3`, will return `[1,x,x^2,x^3]`
                as output, while setting `degree=[1,3]` will return `[x,x^3]` as output.
        """
        self.degree = degree

    ##############
    # Properties #
    ##############

    @property
    def degree(self):
        """Return the degree of the polynomial"""
        return self._degree

    @degree.setter
    def degree(self, degree):
        """Set the degree of the polynomial"""
        # checks
        if isinstance(degree, int):
            degree = range(degree + 1)
        elif isinstance(degree, collections.abc.Iterable):
            for d in degree:
                if not isinstance(d, int):
                    raise TypeError("Expecting the given degrees to be positive integers, but got {}".format(type(d)))
                if d < 0:
                    raise ValueError("Expecting the given degrees to be positive integers, but got {}".format(d))
        else:
            raise TypeError("Expecting the degree to be a positive integer or a list of positive integers.")
        self._degree = degree

    @property
    def size(self):
        """Return the number of exponents. The output vector is then of size = size * len(x)"""
        return len(self.degree)

    ###########
    # Methods #
    ###########

    def reset(self):
        """Reset the polynomial model."""
        pass

    def predict(self, x):
        """Return output polynomial vector"""
        if isinstance(x, np.ndarray):
            return np.concatenate([x**d for d in self.degree])
        elif isinstance(x, torch.Tensor):
            return torch.cat([x**d for d in self.degree])

    def __call__(self, x):
        return self.predict(x)


class Polynomial(object):
    r"""Polynomial model

    The polynomial model is a discriminative deterministic model expressed mathematically as
    :math:`y = f(x) = W \phi(x)`, where :math:`x` is the input vector, :math:`y` is the output vector, :math:`W`
    is the weight matrix, and :math:`\phi` is the polynomial function which returns the transformed input vector.
    This transformed input vector is often of higher dimension, based on the idea that if it is not linear with
    respect to the parameters in the current space, it might be in a higher dimensional space.
    """

    def __init__(self, num_inputs, num_outputs, polynomial_fct):
        r"""
        Initialize the polynomial model: :math:`y = W \phi(x)`

        Args:
            num_inputs (int): dimension of the input vector x
            num_outputs (int): dimension of the output vector y
            polynomial_fct (PolynomialFunction): polynomial function :math:`\phi` to be applied on the input vector x.
        """
        self.phi = polynomial_fct
        num_inputs = num_inputs * self.phi.size
        self.model = torch.nn.Linear(num_inputs, num_outputs, bias=False)
        self._num_parameters = len(self.get_vectorized_parameters())

    ##############
    # Properties #
    ##############

    @property
    def phi(self):
        """Return the polynomial function"""
        return self._phi

    @phi.setter
    def phi(self, function):
        """Set the polynomial function"""
        if not isinstance(function, PolynomialFunction):
            raise TypeError("Expecting the given polynomial function to be an instance of `PolynomialFunction`, "
                            "instead got: {}".format(type(function)))
        self._phi = function

    @property
    def polynomial_function(self):
        """Return the polynomial function"""
        return self._phi

    @polynomial_function.setter
    def polynomial_function(self, function):
        """Set the polynomial function"""
        self.phi = function

    @property
    def input_size(self):
        """Return the input dimension of the model"""
        return self.model.weight.shape[1]

    @property
    def output_size(self):
        """Return the output dimension of the model"""
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
        """Return the input dimension of the model; i.e. len(input_shape)."""
        return len(self.input_shape)

    @property
    def output_dim(self):
        """Return the output dimension of the model; i.e. len(output_shape)."""
        return len(self.output_shape)

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return self._num_parameters

    ##################
    # Static methods #
    ##################

    @staticmethod
    def copy(other):
        """Return another copy of the polynomial model"""
        if not isinstance(other, Polynomial):
            raise TypeError("Trying to copy an object which is not a Polynomial model")
        return copy.copy(other)

    @staticmethod
    def is_parametric():
        """The polynomial model is a parametric model"""
        return True

    @staticmethod
    def is_linear():
        """The polynomial model is linear with respect to its parameters"""
        return True

    @staticmethod
    def is_recurrent():
        """The polynomial model is not recurrent; current outputs do not depend on previous inputs"""
        return False

    @staticmethod
    def is_deterministic():
        """The polynomial model is a deterministic model"""
        return True

    @staticmethod
    def is_probabilistic():     # same as is_stochastic()
        """The polynomial model is not a probabilistic model; it is a deterministic one"""
        return False

    @staticmethod
    def is_discriminative():
        """The polynomial model is a discriminative model."""
        return True

    @staticmethod
    def is_generative():
        """The polynomial model is not a generative model, and thus we can not sample from it"""
        return False

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Returns an iterator over the model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        return self.model.named_parameters()

    def list_parameters(self):
        """Return a list of parameters"""
        return list(self.parameters())

    def hyperparameters(self):
        """Return an iterator over the hyperparameters."""
        for degree in self.phi.degree:
            yield degree

    def named_hyperparameters(self):
        """Return an iterator over the model hyperparameters, yielding both the name and the hyperparameter itself."""
        for idx, degree in enumerate(self.phi.degree):
            yield "degree {}".format(idx), degree

    def list_hyperparameters(self):
        """Return the hyperparameters in the form of a list."""
        return list(self.hyperparameters())

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
        """
        Predict output vector :math:`y` given input vector :math:`x`, using the formula: :math:`y = W \phi(x)`.

        Args:
            x (np.ndarray, torch.Tensor): input vector
            to_numpy (bool): if True, return a np.array

        Returns:
            np.ndarray, torch.Tensor: output vector
        """
        # convert from numpy to pytorch if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # predict the output
        y = self.model(self.phi(x))

        # return the output and convert it if necessary
        if to_numpy:
            if y.requires_grad:
                return y.detach().numpy()
            return y.numpy()
        return y

    def __call__(self, x, to_numpy=True):
        """Predict the output given the input :attr:`x`."""
        return self.predict(x, to_numpy=to_numpy)


# Tests
if __name__ == '__main__':
    # test with numpy
    x = np.array(range(3))
    fct = PolynomialFunction(degree=3)
    model = Polynomial(num_inputs=len(x), num_outputs=2, polynomial_fct=fct)
    print("Polynomial input size: {}".format(model.input_size))
    print("Polynomial output size: {}".format(model.output_size))
    y = model(x)
    print("Polynomial input: {}".format(x))
    print("Polynomial output: {}".format(y))

    # test with pytorch
    x = torch.from_numpy(x).float()
    y = model(x, to_numpy=False)
    print("Polynomial input: {}".format(x))
    print("Polynomial torch output: {}".format(y))
    y = model(x, to_numpy=True)
    print("Polynomial numpy output: {}".format(y))
