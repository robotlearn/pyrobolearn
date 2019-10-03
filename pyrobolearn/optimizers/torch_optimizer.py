# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide wrappers around the torch optimizers.

Compared to the optimizers present in the PyTorch library [1], which require to pass the parameters of the model when
instantiating them, here we can pass the parameters at a later stage.

References:
    [1] https://pytorch.org/docs/stable/optim.html
    [2] https://github.com/sbarratt/torch_cg/tree/master
"""

# Pytorch optimizers
import torch
import torch.nn as nn
import torch.optim as optim

from pyrobolearn.optimizers.optimizer import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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


class Adam(Optimizer):
    r"""Adam Optimizer

    This provides a wrapper around the Adam optimizer [1, 2] where the parameters can be passed at a later stage.
    The documentation is taken from [2].

    References:
        - [1] "Adam: A Method for Stochastic Optimization", Kingma et al., 2014
        - [2] torch.optim: https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
                 max_grad_norm=None, verbose=0, *args, **kwargs):  # 0.5
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): learning rate.
            betas (tuple[float, float]): coefficients used for computing running averages of gradient and its square.
            eps (float): term added to the denominator to improve numerical stability.
            weight_decay (float): weight decay (L2 penalty).
            amsgrad (bool): whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of
                Adam and Beyond".
            max_grad_norm (float, None): the allowed maximum gradient norm. If provided, it will clip the gradients.
            verbose (bool, int): if verbose=2, it will print the gradient norm for each parameter.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(Adam, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

    def reset(self):
        """Reset the optimizer."""
        self.optimizer = None

    def optimize(self, params, loss):
        """
        Optimize the given parameters on the specified loss.

        Args:
            params (list of torch.Tensor): model parameters.
            loss (torch.Tensor): loss value that were computed using the model parameters.
        """
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=self.betas, eps=self.eps,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)

        # optimize
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        if self.verbose > 1:
            total_norm = 0
            for p in params:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2  # item() convert a one element tensor to a scalar
                print("{} - grad norm = {}".format(p, 1./2 * param_norm.item()))
            total_norm = total_norm ** (1. / 2)
            print("Total gradient norm: {}".format(total_norm))

        self.optimizer.step()


class Adadelta(Optimizer):
    r"""Adadelta Optimizer

    This provides a wrapper around the Adadelta optimizer [1, 2] where the parameters can be passed at a later stage.
    The documentation is taken from [2].

    References:
        - [1] "ADADELTA: An Adaptive Learning Rate Method", Zeiler, 2012
        - [2] torch.optim: https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, learning_rate=1., rho=0.9, eps=1e-6, weight_decay=0, max_grad_norm=None, *args, **kwargs):  # 0.5
        """
        Initialize the Adadelta optimizer.

        Args:
            learning_rate (float): learning rate.
            rho (float): coefficient used for computing a running average of squared gradients.
            eps (float): term added to the denominator to improve numerical stability.
            weight_decay (float): weight decay (L2 penalty).
            max_grad_norm (float, None): the allowed maximum gradient norm. If provided, it will clip the gradients.
            verbose (bool, int): if verbose=2, it will print the gradient norm for each parameter.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(Adadelta, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        """
        Optimize the given parameters on the specified loss.

        Args:
            params (list of torch.Tensor): model parameters.
            loss (torch.Tensor): loss value that were computed using the model parameters.
        """
        if self.optimizer is None:
            self.optimizer = optim.Adadelta(params, lr=self.learning_rate, rho=self.rho, eps=self.eps,
                                            weight_decay=self.weight_decay)

        # optimize
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class Adagrad(Optimizer):
    r"""Adagrad Optimizer

    This provides a wrapper around the Adagrad optimizer [1, 2] where the parameters can be passed at a later stage.
    The documentation is taken from [2].

    References:
        - [1] "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", Duchi et al., 2011
        - [2] torch.optim: https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, learning_rate=0.01, learning_rate_decay=0, weight_decay=0, initial_accumumaltor_value=0,
                 max_grad_norm=None, *args, **kwargs):  # 0.5
        """
        Initialize the Adagrad optimizer.

        Args:
            learning_rate (float): learning rate.
            learning_rate_decay (float): learning rate decay.
            weight_decay (float): weight decay (L2 penalty).
            initial_accumulator_value (float): the initial accumulator value.
            max_grad_norm (float, None): the allowed maximum gradient norm. If provided, it will clip the gradients.
            verbose (bool, int): if verbose=2, it will print the gradient norm for each parameter.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(Adagrad, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumumaltor_value
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        """
        Optimize the given parameters on the specified loss.

        Args:
            params (list of torch.Tensor): model parameters.
            loss (torch.Tensor): loss value that were computed using the model parameters.
        """
        if self.optimizer is None:
            self.optimizer = optim.Adagrad(params, lr=self.learning_rate, lr_decay=self.learning_rate_decay,
                                           weight_decay=self.weight_decay,
                                           initial_accumulator_value=self.initial_accumulator_value)

        # optimize
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class RMSprop(Optimizer):
    r"""RMSprop

    This provides a wrapper around the RMSprop optimizer [1, 2, 3] where the parameters can be passed at a later stage.
    The documentation is taken from [3].

    References:
        - [1] "RMSprop:  Divide the gradient by a running average of its recent magnitude" (lecture 6.5), Tieleman and
            Hinton, 2012
        - [2] "Generating Sequences With Recurrent Neural Networks", Graves, 2014
        - [3] torch.optim: https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, learning_rate=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 max_grad_norm=None, *args, **kwargs):  # 0.5
        """
        Initialize the RMSprop optimizer.

        Args:
            learning_rate (float): learning rate.
            alpha (float): smoothing constant.
            eps (float): term added to the denominator to improve numerical stability.
            weight_decay (float): weight decay (L2 penalty).
            momentum (float): momentum factor.
            centered (bool): if True, compute the centered RMSProp, the gradient is normalized by an estimation of
                its variance.
            max_grad_norm (float, None): the allowed maximum gradient norm. If provided, it will clip the gradients.
            verbose (bool, int): if verbose=2, it will print the gradient norm for each parameter.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(RMSprop, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        """
        Optimize the given parameters on the specified loss.

        Args:
            params (list of torch.Tensor): model parameters.
            loss (torch.Tensor): loss value that were computed using the model parameters.
        """
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(params, lr=self.learning_rate, alpha=self.alpha, eps=self.eps,
                                           weight_decay=self.weight_decay, momentum=self.momentum,
                                           centered=self.centered)

        # optimize
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class SGD(Optimizer):
    r"""Stochastic Gradient Descent

    This provides a wrapper around the SGD optimizer [1, 2, 3] where the parameters can be passed at a later stage.
    The documentation is taken from [3].

    References:
        - [1] "A Stochastic Approximation Method", Robbins and Monro, 1951
        - [2] "On the importance of initialization and momentum in deep learning", Sutskever et al., 2013
        - [3] torch.optim: https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, learning_rate=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, max_grad_norm=None,
                 *args, **kwargs):  # 0.5
        """
        Initialize the SGD optimizer.

        Args:
            learning_rate (float): learning rate.
            momentum (float): momentum factor.
            dampening (float): dampening for momentum.
            weight_decay (float): weight decay (L2 penalty).
            nesterov (bool): enables Nesterov momentum.
            max_grad_norm (float, None): the allowed maximum gradient norm. If provided, it will clip the gradients.
            verbose (bool, int): if verbose=2, it will print the gradient norm for each parameter.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(SGD, self).__init__(*args, **kwargs)
        self.optimizer = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        """
        Optimize the given parameters on the specified loss.

        Args:
            params (list of torch.Tensor): model parameters.
            loss (torch.Tensor): loss value that were computed using the model parameters.
        """
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, dampening=self.dampening,
                                       weight_decay=self.weight_decay, nesterov=self.nesterov)

        # optimize
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class CG(Optimizer):
    r"""Conjugate Gradient

    "In mathematics, the conjugate gradient method is an algorithm for the numerical solution of particular systems
    of linear equations, namely those whose matrix is symmetric and positive-definite. The conjugate gradient method
    is often implemented as an iterative algorithm, applicable to sparse systems that are too large to be handled by
    a direct implementation or other direct methods such as the Cholesky decomposition. Large sparse systems often
    arise when numerically solving partial differential equations or optimization problems.

    Suppose we want to solve the system of linear equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`, for the vector
    :math:`\mathbf{x}`, where the known n x n matrix :math:`\mathbf{A}` is symmetric (i.e.,
    :math:`\mathbf{A}^\top = \mathbf{A}`), positive-definite (i.e. :math:`\mathbf{x}^\top \mathbf{Ax} > 0` for all
    non-zero vectors :math:`x \in \mathbb{R}^n`), and real, and :math:`\mathbf{b}` is known as well. We denote the
    unique solution of this system by :math:`\mathbf{x}^*`." [1]

    Notably, this can be used to solve :math:`\mathbf{Hx} = \mathbf{g}` where :math:`\mathbf{H}` is the Hessian
    matrix, and :math:`g` is the gradient.

    References:
        - [1] Conjugate Gradient (Wikipedia): https://en.wikipedia.org/wiki/Conjugate_gradient_method
        - [2] Hessian matrix (Wikipedia): https://en.wikipedia.org/wiki/Hessian_matrix#Use_in_optimization
        - [3] Hessian-Vector products: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
        - [4] Torch CG: https://github.com/sbarratt/torch_cg
    """

    def __init__(self, threshold=1.e-8, max_iters=10, *args, **kwargs):
        """
        Initialize the conjugate gradient.

        Args:
            threshold (float): threshold level.
            max_iters (int): maximum number of iterations.
            *args (list): list of arguments that are given to the parent class `Optimizer`.
            **kwargs (dict): dictionary of arguments that are given to the parent class `Optimizer`.
        """
        super(CG, self).__init__(*args, **kwargs)
        self._threshold = threshold
        self._max_iters = int(max_iters)

    def optimize(self, A, b, x=None):
        """
        Return the solution :math:`x` to the system of linear equations :math:`Ax=b`.

        Notably, this can be used

        Args:
            A (torch.Tensor): square real symmetric and positive-definite matrix.
            b (torch.Tensor): known vector.
            x (torch.Tensor, None): initial solution. If None, it will be the zero vector.

        Returns:
            torch.Tensor: return the solution x to Ax=b.
        """
        if x is None:
            x = torch.zeros_like(b)

        r = b - A.matmul(x)
        p = r
        rr_old = r.t().matmul(r)

        for k in range(self._max_iters):
            Ap = A.matmul(p)
            alpha = rr_old / p.t().matmul(Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = r.t().matmul(r)
            if torch.sqrt(rr_new) < self._threshold:
                break
            p = r + rr_new / rr_old * p
            rr_old = rr_new

        return x
