#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Gaussian Process (GP) learning model.

This file provides the Gaussian Process (GP) model; a non-parametric, discriminative, and probabilistic model.

As for neural networks, several frameworks can be used, such as `GPy` (which uses `numpy`), `GPyTorch` (which uses
`pytorch`), or `GPFlow` (which uses `tensorflow`). We decided to use `GPyTorch` because of the `pytorch` framework
popularity in the research community field, its flexibility, its similarity with numpy (but with automatic
differentiation: autograd), GPU capabilities, and more Pythonic approach.
"""

import copy
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

import numpy as np
import torch
import gpytorch
# import GPy

# from pyrobolearn.models.model import Model

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GP(object):
    r"""Gaussian Process model

    This is a wrapper around the GPyTorch gaussian process models. The Gaussian process is a generalization of
    the multivariate Gaussian distribution. It is a non-parametric, probabilistic, and discriminative model.

    This works by putting a prior distribution on the function:

    .. math:: f ~ GP(0, K(X,X))

    where :math:`K(.,.)` is the kernel matrix where each entry contains :math:`k(x_i, x_j)`, i.e. the kernel function
    evaluated at the corresponding points. As it can be seen the kernel matrix grows with the number of samples.

    See Also:
        - `GPy` (which uses numpy) [4]
        - `GPFlow` (which uses TensorFlow) [5]

    References:
        [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        [2] GPyTorch: https://github.com/cornellius-gp/gpytorch
        [3] GPyTorch examples: https://github.com/cornellius-gp/gpytorch/tree/master/examples
        [4] GPy: https://gpy.readthedocs.io/en/deploy/
        [5] GPFlow: http://gpflow.readthedocs.io/en/latest/intro.html
    """

    def fit(self, *args, **kwargs):
        pass


class GPC(GP):
    r"""Gaussian Process Classification

    References:
        [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        [2] GPyTorch: https://github.com/cornellius-gp/gpytorch
        [3] GPyTorch examples: https://github.com/cornellius-gp/gpytorch/tree/master/examples
    """
    pass


class ExactGPModel(gpytorch.models.ExactGP):
    r"""Create GP prior model.

    That is, it computes :math:`f|X ~ GP(\mu(X), K(X,X))`, or more concretely, it returns the probability
    distribution over the functions given by :math:`N(\mu(x), K(x,x))`. Functions can then be sampled from it;
    :math:`f|X ~ N(\mu(x), K(x,x))`.
    """

    def __init__(self, mean, kernel, likelihood=None, x=None, y=None):
        """
        Initialize the exact Gaussian process model.

        Args:
            mean (gpytorch.means.Mean): mean for the GP prior.
            kernel (gpytorch.kernels.Kernel): kernel function for the GP prior.
            likelihood (None, gpytorch.likelihoods.Likelihood): likelihood pdf.
            x (None, torch.Tensor): training input data.
            y (None, torch.Tensor): training output target data.
        """
        # create likelihood if not already set.
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_inputs=x, train_targets=y, likelihood=likelihood)

        # set mean and kernel covariance function
        self.mean = mean
        self.kernel = kernel

    ##############
    # Properties #
    ##############

    @property
    def mean(self):
        r"""Return the GP prior mean; that is :math:`\mu(x)` from :math:`p(f|x) = N(\mu(x), K(x,x))`."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        r"""Set the GP prior mean; that is :math:`\mu(x)` from :math:`p(f|x) = \mathcal{N}(\mu(x), K(x,x))`."""
        if mean is None:
            mean = gpytorch.means.ConstantMean()
        if not isinstance(mean, gpytorch.means.Mean):
            raise TypeError("Expecting the mean to be an instance of `gpytorch.means.Mean`, got instead "
                            "{}".format(type(mean)))
        self._mean = mean

    @property
    def kernel(self):
        r"""Return the kernel function :math:`K(.,.)` (=prior covariance of the GP)."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        r"""Set the kernel function :math:`K(.,.)` (=prior covariance of the GP)."""
        if kernel is None:
            kernel = gpytorch.kernels.RBFKernel()  # + gpytorch.kernels.WhiteNoiseKernel()
            kernel = gpytorch.kernels.ScaleKernel(kernel)
        if not isinstance(kernel, gpytorch.kernels.Kernel):
            raise TypeError("Expecting the kernel to be an instance of `gpytorch.kernels.Kernel`, got instead "
                            "{}".format(type(kernel)))
        self._kernel = kernel

    ###########
    # Methods #
    ###########

    def forward(self, x):
        r"""Return the prior probability density function :math:`p(f|x) = \mathcal{N}(\cdot | \mu(x), K(x,x))`."""
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return gpytorch.random_variables.GaussianRandomVariable(mean_x, covar_x)


class GPR(GP):
    r"""Gaussian Process Regression

    The Gaussian process is a generalization of the multivariate Gaussian distribution. It is a non-parametric,
    probabilistic, and discriminative model.

    This works by putting a prior distribution on the function:

    .. math:: f|X ~ GP(0, K(X,X))

    where :math:`K(\cdot, \cdot)` is the kernel matrix where each entry contains :math:`k(x_i, x_j)`, i.e. the kernel
    function evaluated at the corresponding points. As it can be seen the kernel matrix grows with the number of
    samples. The kernel function often has hyperparameters :math:`\Phi` that will be optimized.

    The likelihood is given by:

    .. math:: p(y | f) = \mathcal{N}(y | f, \sigma^2 I)

    Learning the hyperparameters of the kernel are carried out by maximizing the marginal log likelihood, which is
    given by:

    .. math::

        \log p(y | X; \Phi) &= \int p(y | f) p(f | X; \Phi) df \\
        \log p(y | X; \Phi) &= -\frac{1}{2} y^\top (K + \sigma^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma^2 I| -
            \frac{n}{2} \log 2\pi

    That is, we optimize the hyperparameters of the kernel function by maximizing the marginal log likelihood:

    .. math::

        \Phi^* = \arg \max_{\Phi} p(Y | X; \Phi)

    The predictive distribution is then carried out by assuming that the observed target values :math:`y` and
    the function values :math:`f^*` at the test locations :math:`X^*` are from the same joint Gaussian distribution.
    By conditioning this distribution with respect to the old dataset :math:`X, y` and the test locations :math:`X^*`,
    we can derive :math:`p(f^* | X, y, X^*)` which is the predictive output distribution given the new data points
    :math:`X^*`.

    Notes:
        * GP takes into account correlations in the input domain but not in the output space
        * The time complexity to learn a GP is :math:`O(N^3)` because of the matrix inversion during training.
        * GMM vs GP:
            * Both are probabilistic models.
            * GMM is a generative semi-parametric model while GP is a discriminative non-parametric model.
            * GMM captures the correlation between the inputs and outputs, while GP only captures correlation in the
                input space.
            * GMR models the variability/correlation between the predicted outputs while the GP provides uncertainty
                on the predicted outputs. The predicted outputs in a GP are independent unless using a heteroscedastic
                GP or a generalized Wishart process is used.

    GPyTorch::

    For most GP regression models, you will need to construct the following GPyTorch objects:
    1. A GP Model (`gpytorch.models.ExactGP`) - This handles most of the inference.
    2. A Likelihood (`gpytorch.likelihoods.GaussianLikelihood`) - This is the most common likelihood used for GP
        regression.
    3. A Mean - This defines the prior mean of the GP. If you don't know which mean to use, a
        `gpytorch.means.ConstantMean` is a good place to start.
    4. A Kernel - This defines the prior covariance of the GP. If you don't know which kernel to use, a
        `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())` is a good place to start.
    5. A MultivariateNormal Distribution (`gpytorch.distributions.MultivariateNormal`) - This is the object used to
        represent multivariate normal distributions.


    References:
        [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        [2] GPy: https://gpy.readthedocs.io/en/deploy/
        [3] GPyTorch: https://github.com/cornellius-gp/gpytorch
        [4] GPFlow: http://gpflow.readthedocs.io/en/latest/intro.html
    """

    def __init__(self,  mean=None, kernel=None, model=None, likelihood=None):
        """
        Initialize the GPR.

        Args:
            mean (None, gpytorch.means.Mean): mean prior. If None, it will be set to `gpytorch.means.ConstantMean()`.
            kernel (None, gpytorch.kernels.Kernel): kernel prior. If None it will be set to
                `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())`
            model (None, gpytorch.module.Module): the prior GP model. If None, it will create `ExactGPModel()`, a GP
                model using the provided mean, kernel, and likelihood.
            likelihood (None, gpytorch.likelihoods.Likelihood): the likelihood pdf. If None, it will use the
                `gpytorch.likelihoods.GaussianLikelihood()`
        """
        # check model
        if model is None:
            self.model = ExactGPModel(mean, kernel, likelihood)
        else:
            self.model = model

        # set model into evaluation mode
        self.eval()

        # set the marginal log likelihood pdf: p(y|x) = \int p(y|f,x) p(f|x) df
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_prob, self.prior)

    ##############
    # Properties #
    ##############

    @property
    def model(self):
        """Return the GP model; i.e. the prior probability density function p(f|x)."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the GP model; i.e. the prior probability density function p(f|x)."""
        if not isinstance(model, gpytorch.module.Module):
            raise TypeError("Expecting the GP model to be an instance of `gpytorch.module.Module`, got instead "
                            "{}".format(type(model)))
        self._model = model

    # alias
    prior = model

    @property
    def likelihood_prob(self):
        """Return the likelihood probability density function p(y|f,x)."""
        return self.model.likelihood

    @likelihood_prob.setter
    def likelihood_prob(self, likelihood):
        r"""Set the likelihood probability density function p(y|f,x)."""
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise TypeError("Expecting the likelihood to be an instance of `gpytorch.likelihood.Likelihood`, got "
                            "instead {}".format(type(likelihood)))
        self.model.likelihood = likelihood

    @property
    def log_marginal_likelihood_prob(self):
        r"""Return the log marginal log likelihood pdf: log p(y|x)."""
        return self.mll

    @property
    def mean(self):
        r"""Return the GP prior mean; that is :math:`\mu(x)` from :math:`p(f|x) = N(\mu(x), K(x,x))`."""
        return self.model.mean

    @mean.setter
    def mean(self, mean):
        r"""Set the GP prior mean; that is :math:`\mu(x)` from :math:`p(f|x) = \mathcal{N}(\mu(x), K(x,x))`."""
        if mean is None:
            mean = gpytorch.means.ConstantMean()
        if not isinstance(mean, gpytorch.means.Mean):
            raise TypeError("Expecting the mean to be an instance of `gpytorch.means.Mean`, got instead "
                            "{}".format(type(mean)))
        self.model.mean = mean

    @property
    def kernel(self):
        r"""Return the kernel function :math:`K(.,.)` (=prior covariance of the GP)."""
        return self.model.kernel

    @kernel.setter
    def kernel(self, kernel):
        r"""Set the kernel function :math:`K(.,.)` (=prior covariance of the GP)."""
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())
        if not isinstance(kernel, gpytorch.kernels.Kernel):
            raise TypeError("Expecting the kernel to be an instance of `gpytorch.kernels.Kernel`, got instead "
                            "{}".format(type(kernel)))
        self.model.kernel = kernel

    @property
    def dim(self):
        """Return the dimension of the kernel"""
        return self.kernel.dim

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other, deep=True):
        """copy the other GP"""
        if not isinstance(other, GPR):
            raise TypeError("Expecting a GPR model.")
        if deep:
            return copy.deepcopy(other)
        return copy.copy(other)

    @staticmethod
    def is_parametric():
        """The Gaussian process is a non parametric model."""
        return False

    @staticmethod
    def is_linear():
        """The Gaussian process does not have any parameters and thus no linear parameters"""
        return False

    @staticmethod
    def is_recurrent():
        """The Gaussian process is not a recurrent model"""
        return False

    @staticmethod
    def is_probabilistic():
        """The Gaussian process is a probabilistic model"""
        return True

    @staticmethod
    def is_discriminative():
        """The Gaussian process is a discriminative model which predicts :math:`p(y|x)`"""
        return True

    @staticmethod
    def is_generative():
        """The Gaussian process is not a generative model, and thus we can not sample from it"""
        # TODO: actually we can sample a function from it given the initial data (in kernel matrix)
        return False

    @staticmethod
    def load(filename):
        """
        Load a model from memory.

        Args:
            filename (str): file that contains the model.
        """
        return pickle.load(filename)

    @staticmethod
    def _convert_to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x

    @staticmethod
    def _convert_to_numpy(x):
        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                return x.detach().numpy()
            return x.numpy()
        return x

    @staticmethod
    def _convert(x, to_numpy=True):
        if to_numpy:
            return GPR._convert_to_numpy(x)
        return x

    ###########
    # Methods #
    ###########

    def train(self, mode=True):
        """Set the model into training mode."""
        if mode:
            self.model.train(True)
            self.likelihood_prob.train(True)
        else:
            self.model.train(False)
            self.likelihood_prob.train(False)

    def eval(self):
        """Set the mode into evaluation mode."""
        self.model.eval()
        self.likelihood_prob.eval()

    def parameters(self):
        return self.model.parameters()

    def hyperparameters(self):
        """Return an iterator over the hyperparameters"""
        return self.model.hyperparameters()

    def named_hyperparameters(self):
        """Return an iterator over the model parameters, yielding both the name and the parameter itself."""
        return self.model.named_hyperparameters()

    def likelihood(self, x, y, to_numpy=False):
        r"""Evaluate the likelihood p(y|f,x)."""
        likelihood = torch.exp(self.log_likelihood(x, y, to_numpy=False))
        return self._convert(likelihood, to_numpy=to_numpy)

    def log_likelihood(self, x, y, to_numpy=False):
        r"""Evaluate the log likelihood: log p(y|f,x)."""
        x = self._convert_to_torch(x)
        y = self._convert_to_torch(y)
        f = self.model(x)
        log_likelihood = self.likelihood_prob.log_probability(f, y)
        return self._convert(log_likelihood, to_numpy=to_numpy)

    def marginal_likelihood(self, x, y, to_numpy=False):
        r"""Evaluate the marginal likelihood: p(y|x)."""
        ml = torch.exp(self.log_marginal_likelihood(x, y, to_numpy=False))
        return self._convert(ml, to_numpy=to_numpy)

    def log_marginal_likelihood(self, x, y, to_numpy=False):
        r"""Evaluate the log marginal likelihood: log p(y|x)."""
        x = self._convert_to_torch(x)
        y = self._convert_to_torch(y)
        f = self.model(x)
        mll = self.mll(f, y)
        return self._convert(mll[0], to_numpy=to_numpy)

    def fit(self, x, y, num_iters=100, tolerance=1e-5, optimizer=None, verbose=False):
        r"""Fit the input and output data; find optimal model hyperparameters.

        Args:
            x (torch.Tensor, np.array): input data.
            y (torch.Tensor, np.array): output data.
            num_iters (int): number of iterations to optimize the marginal likelihood.
            tolerance (float): termination tolerance on function value/parameter changes (default: 1e-5).
            optimizer (None, torch.optim.Optimizer): optimizer to use to optimize the marginal likelihood. By default,
                it will use L-BFGS.
            verbose (bool): if True, it will print information during the optimization.
        """
        # check input and output data
        x = self._convert_to_torch(x)
        y = self._convert_to_torch(y)

        # set training data
        # self.model.set_train_data(x, y)
        self.model = ExactGPModel(self.mean, self.kernel, self.likelihood_prob, x, y)  # TODO

        # set into training mode
        self.train(mode=True)

        # define optimizer
        need_closure = False
        if optimizer is None or (isinstance(optimizer, str) and optimizer.lower() == 'lbfgs'):
            optimizer = torch.optim.LBFGS([{'params': self.parameters()}], max_iter=num_iters,
                                          tolerance_change=tolerance)
            need_closure = True

            def closure():
                optimizer.zero_grad()
                output = self.model(x)
                loss = -self.mll(output, y)
                loss.backward()
                return loss

        elif isinstance(optimizer, str) and optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=0.1)

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Expecting the optimizer to be an instance of `torch.optim.Optimizer`.")

        # optimize
        for i in range(num_iters):
            # zero gradients from previous iteration
            optimizer.zero_grad()

            # predict output from model p(f|x)
            output = self.model(x)

            # compute loss
            loss = - self.mll(output, y)

            # call backward on the loss to fill the gradients
            loss.backward()

            # print info if specified
            if verbose:
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i + 1, num_iters, loss.item(),
                #       self.model.kernel.base_kernel.lengthscale.item(), self.model.likelihood.noise.item()))
                print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iters, loss.item()))

            # perform a step with the optimizer
            if need_closure:
                optimizer.step(closure)
            else:
                optimizer.step()

        # set into evaluation mode (=predictive posterior mode)
        self.eval()

    def predict(self, x, to_numpy=True):
        r"""
        Predict mean output array :math:`\mu(y)` given input array :math:`x`. That is, it returns the mean of the
        predictive posterior distribution :math:`E[p(y|x)]`.

        Args:
            x (np.ndarray, torch.Tensor): input array
            to_numpy (bool): if True, return a np.array

        Returns:
            np.ndarray, torch.Tensor: output mean array
        """
        x = self._convert_to_torch(x)

        # compute p(f|x)
        f = self.model(x)

        # compute p(y|f,x)
        y = self.likelihood_prob(f)

        # return mean
        if to_numpy:
            return self._convert_to_numpy(y.mean())
        return y.mean()

    def predict_prob(self, x, to_numpy=True):
        r"""
        Predict p(y|x) by returning the mean and the covariance arrays.

        Args:
            x (np.ndarray, torch.Tensor): input array
            to_numpy (bool): if True, return a np.array

        Returns:
            np.ndarray, torch.Tensor: output mean array
            np.ndarray, torch.Tensor: output covariance array
        """
        x = self._convert_to_torch(x)

        # compute p(f|x)
        f = self.model(x)

        # compute p(y|f,x)
        y = self.likelihood_prob(f)

        # return mean and covariance
        if to_numpy:
            return self._convert_to_numpy(y.mean()), self._convert_to_numpy(y.var())  # y.covar())
        return y.mean(), y.var()  # y.covar()

    def forward(self, x):
        r"""
        Return the predictive distribution p(y|x).

        Args:
            x (np.ndarray, torch.Tensor): input array

        Returns:
            gpytorch.random_variables.GaussianRandomVariable: multivariate normal (Gaussian) distribution
        """
        x = self._convert_to_torch(x)

        # compute p(f|x)
        f = self.model(x)

        # return p(y|f,x)
        return self.likelihood_prob(f)

    def sample(self, x, num_samples=1, to_numpy=True):
        """Sample the function vector from the GP; i.e. f ~ p(f|x)."""
        x = self._convert_to_torch(x)
        f = self.model(x)
        if to_numpy:
            return self._convert_to_numpy(f.sample(num_samples))
        return f.sample(num_samples)

    #############
    # Operators #
    #############

    def __str__(self):
        """Return name of class"""
        return self.__class__.__name__


# class GPRTorch(GPR):
#     r"""Gaussian Process Regression using GPyTorch
#
#     This provides a wrapper around the GPyTorch module.
#
#     References:
#         [1] "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration", Gardner et al., 2018
#         [2] GPyTorch: https://github.com/cornellius-gp/gpytorch
#         [3] GPyTorch Examples: https://github.com/cornellius-gp/gpytorch/tree/master/examples
#     """
#
#     def __init__(self, model):
#         super(GPRTorch, self).__init__(model)
#
#     def _predict(self, x=None):
#         return self.model(x)


# class GPRy(GPR):
#     r"""Gaussian Process Regression using GPy.
#
#     References:
#         [1] "GPy: A Gaussian process framework in python", Sheffield, 2014
#         [2] GPy: https://github.com/SheffieldML/GPy
#     """
#
#     def __init__(self, model):
#         super(GPRy, self).__init__(model)
#
#     def _predict(self, x=None, full_cov=False):
#         return self.model.predict(x, full_cov)[0]


# TESTS
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pyrobolearn.utils.converter import torch_to_numpy

    # create input and output data
    x = torch.linspace(0, 1, 100)
    y = torch.sin(x * (2 * np.pi)) + torch.randn(x.size()) * 0.2

    x, y = x.numpy(), y.numpy()

    # plot true data
    plt.plot(x, y, 'x')

    # create GPR
    model = GPR()

    # plot prior possible functions
    f = model.sample(x, num_samples=10, to_numpy=True)
    plt.plot(x, f.T)

    # compute log likelihoods
    print("\nBefore training:")
    print("Log likelihood: {}".format(model.log_likelihood(x, y, to_numpy=True)))
    print("Log marginal likelihood: {}".format(model.log_marginal_likelihood(x, y, to_numpy=True)))

    # fit the data
    optimizer = 'adam'  # 'lbfgs'
    model.fit(x, y, num_iters=100, optimizer=optimizer, verbose=True)

    # compute log likelihoods
    print("\nAfter training:")
    print("Log likelihood: {}".format(model.log_likelihood(x, y, to_numpy=True)))
    print("Log marginal likelihood: {}".format(model.log_marginal_likelihood(x, y, to_numpy=True)))

    # sample function and plot it
    f = model.sample(x, num_samples=1, to_numpy=True)
    plt.plot(x, f.T, 'k', linewidth=2.)

    # predict prob
    x_test = torch.linspace(0, 1, 51).numpy()
    mean_y, var_y = model.predict_prob(x_test, to_numpy=True)
    std_y = np.sqrt(var_y)
    plt.plot(x_test, mean_y, 'b')
    plt.fill_between(x_test, mean_y-2*std_y, mean_y+2*std_y, facecolor='green', alpha=0.5)
    plt.ylim([-3, 3])
    plt.show()

    # Another way to predict
    pred = model.forward(x_test)
    lower, upper = pred.confidence_region()

    plt.plot(x, y, 'k*')
    plt.plot(x_test, torch_to_numpy(pred.mean()), 'b')
    plt.fill_between(x_test, torch_to_numpy(lower), torch_to_numpy(upper), alpha=0.5)
    plt.ylim([-3, 3])
    plt.show()
