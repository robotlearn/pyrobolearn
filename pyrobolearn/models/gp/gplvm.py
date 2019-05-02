#!/usr/bin/env python
"""Provide the Gaussian Process Latent Variable Model (GPLVM)

This file provides the GPLVM and shared GPLVM.
"""

import torch
import gpytorch

from pyrobolearn.models.gp.gp import GPR


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GPLVM(GPR):
    r"""Gaussian Process Latent Variable Model (GPLVM)

    The GPLVM is a non-linear, probabilistic model which reduces the dimensional of the original observational space
    by projecting it to a latent space. This can be seen as one of the non-linear version of probabilistic PCA.

    The optimization carried out by the GPLVM is similar to the one undertaken by the GP regression, instead that, in
    addition to optimize the kernel hyperparameters, we also optimize the latent variables. This is mathematically
    formulated by:

    .. math::

        {X^*, \Phi^*} = \arg \max_{X, \Phi} p(Y|X; \Phi)

    where :math:`X` are the latent variables, :math:`\Phi` are the kernel hyperparameters, :math:`Y` are the
    variables that are observed, and :math:`p(Y|X; \Phi)` is the marginal likelihood.
    Note that the initialization of the latent space of the GP-LVM can have drastic consequences on the results;
    which can be good or bad depending on the initialization scheme. Different initialization schemes can be used
    such as random, or PCA.

    A GPLVM provides a smooth mapping from the latent space to the observational space, however the converse is not
    true. In order to preserve local distances and have a smooth mapping from the observational space to the latent
    space, back constraints are often used, which is given by:

    .. math::

        {W^*, \Phi^*} = \arg \max_{W, \Phi} p(Y|X; \Phi)

    where :math:`X = g(Y; W)` with :math:`g` is the back-constraint function parametrized by the weights :math:`W`,
    which are learned during the optimization process. This often has the effect of constraining the latent space (and
    thus the latent parameters) making it more robust to overfitting.

    References:
        [1] "Gaussian Process Latent Variable Models for Visualisation of High Dimensional Data", Lawrence, 2004
        [2] "Probabilistic Non-linear Principal Component Analysis with Gaussian Process Latent Variable Models",
            Lawrence, 2005
        [3] "Local distance preservation in the GP-LVM through back constraints", Lawrence et al., 2006
    """

    def __init__(self, latent_dim, mean=None, kernel=None, model=None, likelihood=None):
        """
        Initialize the GPLVM.

        Args:
            latent_dim (int): latent dimensionality.
            mean (None, gpytorch.means.Mean): mean prior. If None, it will be set to `gpytorch.means.ConstantMean()`.
            kernel (None, gpytorch.kernels.Kernel): kernel prior. If None it will be set to
                `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())`
            model (None, gpytorch.module.Module): the prior GP model. If None, it will create `ExactGPModel()`, a GP
                model using the provided mean, kernel, and likelihood.
            likelihood (None, gpytorch.likelihoods.Likelihood): the likelihood pdf. If None, it will use the
                `gpytorch.likelihoods.GaussianLikelihood()`
        """
        super(GPLVM, self).__init__(mean=mean, kernel=kernel, model=model, likelihood=likelihood)
        self.X = None
        latent_dim = int(latent_dim)
        if latent_dim <= 0:
            raise ValueError("Expecting the latent dimension to be bigger than 0, instead got: {}".format(latent_dim))
        self.latent_dim = latent_dim

    @property
    def is_latent(self):
        """The GPLVM is a latent model."""
        return True

    # TODO: improve the below function
    def fit_latent(self, Y, init='pca', num_iters=100, tolerance=1e-5, optimizer=None, verbose=False):
        """Fit the given data by maximizing the marginal likelihood.

        Args:
            Y (torch.Tensor, np.array): observational data.
            init (str, torch.nn.Module): initialization scheme or back-constraint function. If a string, it is the
                initialization scheme. Currently, you can choose between 'pca' and 'random'. If torch.nn.Module, it
                is the back-constraint function which has some parameters to optimize.
            num_iters (int): number of iterations to optimize the marginal likelihood.
            tolerance (float): termination tolerance on function value/parameter changes (default: 1e-5).
            optimizer (None, torch.optim.Optimizer): optimizer to use to optimize the marginal likelihood. By default,
                it will use L-BFGS.
            verbose (bool): if True, it will print information during the optimization.
        """
        # convert observation data to torch.tensor
        Y = self._convert_to_torch(Y)

        # initialize the latent space
        if isinstance(init, torch.nn.Module):  # back-constraint function
            raise NotImplementedError
        elif isinstance(init, str):
            init = init.lower()
            if init == 'random':
                X = torch.rand(Y.shape[0], self.latent_dim)
            elif init == 'pca':
                X = Y - torch.mean(Y, dim=0).expand_as(Y)
                U, S, V = torch.svd(X)
                X = torch.mm(X, V[:, :self.latent_dim])
            else:
                raise NotImplementedError("Currently, only the 'random' or 'pca' initialization schemes have been "
                                          "implemented")
            X.requires_grad = True
            X = torch.nn.Parameter(X)

        self.X = X

        # fit the data
        self.fit(X, Y, num_iters=num_iters, tolerance=tolerance, optimizer=optimizer, verbose=verbose)


class SGPLVM(GPLVM):
    r"""Shared Gaussian Process Latent Variable Model (SGPLVM)

    With :math:`k` observational spaces sharing the same latent latent space, the following marginal likelihood to
    maximize is then given by:

    .. math::

        {X^*, {\Phi^*_i}_{i=1}^k} = \arg \max_{X, {\Phi_i}_{i=1}^k}  p({Y_i}_{i=1}^k | X; {\Phi_i}_{i=1}^k)

    Back-constraints can be applied on one of the observational space :math:`Y_i`, such that:

    .. math::

        {W^*, {\Phi^*_i}_{i=1}^k} = \arg \max_{W, {\Phi_i}_{i=1}^k}  p({Y_i}_{i=1}^k | X; {\Phi_i}_{i=1}^k)

    where :math:`X = g(Y_i, W)` with :math:`g` is the back-constraint function parametrized by the weights :math:`W`.

    References:
        [1] "Shared gaussian process latent variables models" (PhD thesis), Ek, 2009
    """

    def __init__(self, latent_dim, mean=None, kernel=None, model=None, likelihood=None):
        """
        Initialize the shared GPLVM.

        Args:
            latent_dim (int): latent dimensionality.
            mean (None, gpytorch.means.Mean): mean prior. If None, it will be set to `gpytorch.means.ConstantMean()`.
            kernel (None, gpytorch.kernels.Kernel): kernel prior. If None it will be set to
                `gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.WhiteNoiseKernel())`
            model (None, gpytorch.module.Module): the prior GP model. If None, it will create `ExactGPModel()`, a GP
                model using the provided mean, kernel, and likelihood.
            likelihood (None, gpytorch.likelihoods.Likelihood): the likelihood pdf. If None, it will use the
                `gpytorch.likelihoods.GaussianLikelihood()`
        """
        super(SGPLVM, self).__init__(latent_dim=latent_dim, mean=mean, kernel=kernel, model=model,
                                     likelihood=likelihood)


class HGPLVM(GPLVM):
    r"""Hierarchical Gaussian Process Latent Variable Model.

    References:
        [1] "Hierarchical Gaussian Process Latent Variable Models", Lawrence et al., 2007
    """
    pass
