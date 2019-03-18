#!/usr/bin/env python
"""Provide wrappers around the torch optimizers.

Compared to the optimizers present in the PyTorch library [1], which require to pass the parameters of the model when
instantiating them, here we can pass the parameters at a later stage.

References:
    [1] https://pytorch.org/docs/stable/optim.html
"""

# Pytorch optimizers
import torch.nn as nn
import torch.optim as optim

from optimizer import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# class PyTorchOpt(Optimizer):
#     r"""PyTorch Optimizers
#
#     This is a wrapper around the optimizers from pytorch.
#     """
#
#     def __init__(self, model, losses, hyperparameters):
#         super(PyTorchOpt, self).__init__(model, losses, hyperparameters)
#
#     def add_constraint(self):
#         # it will add a constraint as the augmented lagrangian
#         pass


class Adam(object):
    r"""Adam Optimizer

    References:
        [1] "Adam: A Method for Stochastic Optimization", Kingma et al., 2014
    """

    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.max_grad_norm = max_grad_norm

    def reset(self):
        self.optimizer = None

    def optimize(self, params, loss):
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=self.betas, eps=self.eps,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class Adadelta(object):
    r"""Adadelta Optimizer

    References:
        [1] "ADADELTA: An Adaptive Learning Rate Method", Zeiler, 2012
    """

    def __init__(self, learning_rate=1., rho=0.9, eps=1e-6, weight_decay=0, max_grad_norm=None): #0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.Adadelta(params, lr=self.learning_rate, rho=self.rho, eps=self.eps,
                                            weight_decay=self.weight_decay)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class Adagrad(object):
    r"""Adagrad Optimizer

    References:
        [1] "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", Duchi et al., 2011
    """

    def __init__(self, learning_rate=0.01, learning_rate_decay=0, weight_decay=0, initial_accumumaltor_value=0,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumumaltor_value
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.Adagrad(params, lr=self.learning_rate, lr_decay=self.learning_rate_decay,
                                           weight_decay=self.weight_decay,
                                           initial_accumulator_value=self.initial_accumulator_value)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class RMSprop(object):
    r"""RMSprop

    References:
        [1] "RMSprop:  Divide the gradient by a running average of its recent magnitude" (lecture 6.5), Tieleman and
            Hinton, 2012
        [2] "Generating Sequences With Recurrent Neural Networks", Graves, 2014
    """

    def __init__(self, learning_rate=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(params, lr=self.learning_rate, alpha=self.alpha, eps=self.eps,
                                           weight_decay=self.weight_decay, momentum=self.momentum,
                                           centered=self.centered)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class SGD(object):
    r"""Stochastic Gradient Descent

    References:
        [1] "A Stochastic Approximation Method", Robbins and Monro, 1951
        [2] "On the importance of initialization and momentum in deep learning", Sutskever et al., 2013
    """

    def __init__(self, learning_rate=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 max_grad_norm=None): #0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, dampening=self.dampening,
                                       weight_decay=self.weight_decay, nesterov=self.nesterov)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()
