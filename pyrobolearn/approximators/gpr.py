#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define Gaussian Process function approximator.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.gp import GPR


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GPRApproximator(Approximator):
    r"""Gaussian Process Regression Approximator

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
        - [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        - [2] GPy: https://gpy.readthedocs.io/en/deploy/
        - [3] GPyTorch: https://github.com/cornellius-gp/gpytorch
        - [4] GPFlow: http://gpflow.readthedocs.io/en/latest/intro.html
    """

    def __init__(self, inputs, outputs, mean=None, kernel=None, model=None, likelihood=None,
                 preprocessors=None, postprocessors=None):
        """
        Initialize the Gaussian process regression approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of Action/State)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            mean (None, gpytorch.means.Mean): mean prior. If None, it will be set to `gpytorch.means.ConstantMean()`.
            kernel (None, gpytorch.kernels.Kernel): kernel prior. If None it will be set to
              `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())`
            model (None, gpytorch.module.Module): the prior GP model. If None, it will create `ExactGPModel()`, a GP
              model using the provided mean, kernel, and likelihood.
            likelihood (None, gpytorch.likelihoods.Likelihood): the likelihood pdf. If None, it will use the
              `gpytorch.likelihoods.GaussianLikelihood()`.
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
              to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
              processors before being returned.
        """
        # create inner model
        num_inputs, num_outputs = self._size(inputs), self._size(outputs)
        model = GPR(mean=mean, kernel=kernel, model=model, likelihood=likelihood)

        # call parent class
        super(GPRApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                              postprocessors=postprocessors)
