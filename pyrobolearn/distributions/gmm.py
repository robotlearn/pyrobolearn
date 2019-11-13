#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Gaussian Mixture Model and Gaussian Mixture Regression

This file provides the Gaussian Mixture Model implemented in PyTorch, and uses the Gaussian model defined in the
`gaussian.py` file. Gaussian Mixture Regression is achieved by conditioning the GMM to some input.
"""

import numpy as np
import torch
try:
    import cPickle as pickle
except ImportError as e:
    import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from pyrobolearn.distributions.gaussian import Gaussian


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GMM(torch.distributions.Distribution):
    r"""Gaussian Mixture Model

    This class described the Gaussian Mixture Model (GMM); a semi-parametric, probabilistic and generative model [1,2].
    In robotics, for instance, this is often used to model trajectories by jointly encoding the time and state
    (position and velocity) [3,4,5,6].

    It is mathematically described by:

    .. math:: p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mu_k, \Sigma_k)

    where :math:`K` is the number of components, :math:`\pi_k` are prior probabilities (that is
    :math:`0 \leq \pi_k \leq 1`) that sums to 1 (i.e. :math:`\sum_{k=1}^K \pi_k = 1`),
    :math:`\mathcal{N}(\mu_k, \Sigma_k)` is the multivariate Gaussian (aka Normal) distribution, with mean
    :math:`\mu_k` and covariance :math:`\Sigma_k`. The priors, means, and covariances are grouped to form the
    parameter set :math:`\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K`.


    Learning from data:
    -------------------

    There are three main ways to learn the parameters: maximum likelihood estimate (MLE), maximum a posteriori
    estimate (MAP), and bayesian inference (using variational inference). Here, we will focus on MLE.

    Given a dataset :math:`X \in \mathbb{R}^{N \times D}`, the log-likelihood of the GMM is given by:

    .. math::

        \mathcal{L}(\theta) &= \log p(X | \theta) = \log p(X | \pi, \mu, \Sigma) \\
                            &= \sum_{n=1}^N \log \sum_{k=1}^K \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)

    The summation inside the logarithm in the above loss does not allow for a closed-form solution. We thus turn our
    attention to an iterative algorithm that maximizes this last one.

    The Expectation-Maximization (EM) algorithm [1,2] allows to find the maximum likelihood estimate for models having
    latent variables. This algorithm consists of 4 main steps:
    1. Initialize the parameters :math:`\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K`
    2. Expectation step: evaluate the posterior :math:`p(Z | X, \theta_{old})` while fixing the parameters.
    3. Maximization step: maximize the expected value of the complete-data log-likelihood under the posterior
        distribution of the latent variables (found during the Expectation step). That is,
        :math:`\max_\theta Q(\theta, \theta_{old}) = \max_\theta \sum_{Z} p(Z | X, \theta_{old}) \log p(X,Z | \theta)`.
    4. Evaluate the log-likelihood loss, and check if it converged. If it didn't, go back to step 2.

    To be more specific, the EM algorithm alternatively computes a lower bound on the log-likelihood for the current
    parameters, and then maximize this bound to obtain the new parameter values (see [1], sec 9.4 for more details).
    This results in the above algorithm.

    Few notes with respect to the EM algorithm:
    * this guarantees an improvement over the but the initialization is quite important. In the literature, we can
        often initialize it using the K-means algorithm.
    * while other learning algorithms such as gradient ascent could be used, one of the major problem is that they
        do not enforce constraints on the priors and covariance matrices during the optimization.

    For other variants of the EM algorithm, please refer to [2], section 11.4.9.


    Gaussian Mixture Regression (i.e. conditioned GMM):
    --------------------------------------------------

    Gaussian Mixture Regression [3,4] consists to condition the GMM (that models the joint distribution over the input
    and output variables :math:`p(x^I, x^O)`) on a part of the variables (for instance, the input variables
    :math:`p(x^O | x^I`). Let's :math:`x = [x^I, x^O]`, :math:`\mu_k = [\mu_k^I, \mu_k^O]`, and :math:`\Sigma_k =
    \left[ \begin{array}{cc} \Sigma_k^I & \Sigma_k^{IO} \\ \Sigma_k^{OI} & \Sigma_k^O \end{array} \right]`, where
    :math:`I` and :math:`O` are the superscripts to refer the input and output respectively.

    .. math::

        p(x^O | x^I) &= \sum_{k=1}^K p(z_k=1 | x^I) p(x^O | x^I, z_k=1) \\
                     &= \sum_{k=1}^K r_k(x^I) \mathcal{N}(\hat{\mu}_k^O(x^I), \hat{\Sigma}_k^O)

    where :math:`r_k(x^I) = \frac{\pi_k \mathcal{N}(x^I|\mu_k^I, \Sigma_k^I)}{\sum_{j=1}^{K} \pi_j
    \mathcal{N}(x^I|\mu_j^I,\Sigma_j^I)}` are the responsibilities, :math:`\hat{\mu}_k^O(x^I) =
    \mu_k^O + \Sigma_k^{OI} \Sigma_k^I^{-1} (x^I - \mu_k^I)` and :math:`\hat{\Sigma}_k^O = \Sigma_k^O -
    \Sigma_k^{OI} (\Sigma_k^I)^{-1} \Sigma_k^{IO}` are the resulting conditioned means and covariances,
    respectively.

    This results in another GMM, which can be approximated by a simple Gaussian (see [4] for more info, or the
    documentation of the corresponding method: `approximate_by_single_gaussian`):

    .. math::

        p(x^O | x^I) \approx \mathcal{N}(x^O | \hat{\mu}^O(x^I), \hat{\Sigma}^O(x^I))

    where :math:`\hat{\mu}^O(x^I) = \sum_{k=1}^K r_k(x^I) \hat{\mu}_k^O(x^I)` and :math:`\hat{\Sigma}^O(x^I) =
    \sum_{k=1}^K r_k(x^I) (\hat{\Sigma}_k^O + \hat{\mu}_k^O(x^I) \hat{\mu}_k^O(x^I)^T) - \hat{\mu}^O(x^I)
    \hat{\mu}^O(x^I)^T`.


    Other miscellaneous information:
    --------------------------------

    The conjugate prior of the GMM is the Dirichlet process.


    References:
        [1] "Pattern Recognition and Machine Learning" (chap 2, 3, 9, and 10), Bishop, 2006
        [2] "Machine Learning: a Probabilistic Perspective" (chap 3 and 11), Murphy, 2012
        [3] "Robot Programming by Demonstration: a Probabilistic Approach" (chap 2), Calinon, 2009
        [4] "A Tutorial on Task-Parameterized Movement Learning and Retrieval", Calinon, 2015
        [5] "Programming by Demonstration on Riemannian Manifolds" (PhD thesis, chap 1 and 2), Zeerstraten, 2017
        [6] "Learning Control", Calinon et al., 2018

    The code was inspired by the following codes:
    - `gaussian.py`: defines the Gaussian distribution
    - `sklearn.mixture.gmm` and `sklearn.mixture.dpgmm`: http://scikit-learn.org/stable/modules/mixture.html
    - `gmr`: https://github.com/AlexanderFabisch/gmr
    - `pybdlib`: https://gitlab.idiap.ch/rli/pbdlib-python/tree/master/pbdlib
    - `riepybdlib.statistics`: https://gitlab.martijnzeestraten.nl/martijn/riepybdlib
    """

    def __init__(self, num_components=1, priors=None, means=None, covariances=None, gaussians=None, seed=None,
                 dimensionality=None, manifold=None):
        """
        Initialize the Gaussian Mixture Model (GMM).

        Args:
            num_components (int): the number of components/gaussians (this argument should be provided if
                no priors, means, covariances, or gaussians are provided)
            priors (list/tuple of floats, None): prior probabilities (they have to be positives). If not provided,
                it will be a uniform distribution.
            means (list of torch.Tensor[D], None): list of means
            covariances (list of torch.Tensor[D,D], None): list of covariances
            gaussians (list of Gaussian, None): list of gaussians. If provided, the `means` and `covariances`
                parameters don't have to be provided.
            seed (int): random seed. Useful when sampling and for EM algo.
            dimensionality (int, None): dimensionality of the data (this can be inferred during the training process)
            manifold (None): By default, it is the Euclidean space.
        """
        super(GMM, self).__init__()

        # variables: number of components, dimensionality, and the number of data
        self.K = num_components
        self.N = 0

        # set seed
        self.seed = seed

        # set gaussians
        self._gaussians = []
        if gaussians is None:
            if means is not None and covariances is not None:
                self._gaussians = [Gaussian(mean=mean, covariance=cov) for mean, cov in zip(means, covariances)]
                self.K = len(self._gaussians)
        else:
            self._gaussians = gaussians
            self.K = len(self._gaussians)

        # set priors
        self.priors = priors

        # check if the priors and gaussians have the same number of components
        if priors is not None and gaussians is not None:
            if len(priors) != len(gaussians):
                raise ValueError("The number of priors and gaussians are differents")

    ##############
    # Properties #
    ##############

    @property
    def seed(self):
        """Return the seed"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the random seed"""
        self._seed = seed
        if self._seed is not None:
            torch.manual_seed(self._seed)

    @property
    def num_components(self):
        """Return the number of components, i.e. the number of Gaussians"""
        return self.K

    # alias
    size = num_components

    @property
    def dimensionality(self):
        """Return the dimensionality of the mean"""
        if len(self._gaussians) > 0:
            return self._gaussians[0].dim
        return 0

    # alias
    dim = dimensionality

    @property
    def num_parameters(self):
        """Return the number of free parameters"""
        D = self.dimensionality
        return (self.K - 1) + self.K * (D + 1./2 * D * (D + 1))

    @property
    def num_data(self):
        """Return the number of data points"""
        return self.N

    @property
    def priors(self):
        r"""Return the priors :math:`\pi_k`"""
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Set the priors"""
        if priors is not None:
            priors = torch.tensor(priors, dtype=torch.float)

            # check shape
            if len(priors.shape) != 1:
                raise ValueError("Expecting 1d array for the priors, instead got a shape of {}".format(priors.shape))

            # check if the priors are positives
            if not torch.all(priors >= 0):
                raise ValueError("Some priors are not positives")

            # re-normalize just in case
            if len(priors) > 0:
                priors /= torch.sum(priors)

            # set the priors and the number of components
            self._priors = priors
            self.K = len(self._priors)
        else:
            if self.num_components > 0:
                self._priors = torch.ones(self.num_components) / self.num_components
            else:
                self._priors = torch.tensor([])

    @property
    def gaussians(self):
        r"""Return the Gaussian distributions :math:`\mathcal{N}(\mu_k, \Sigma_k)`"""
        return self._gaussians

    @property
    def means(self):
        """Return the means (shape: KxD): the mean of each Gaussian"""
        return torch.tensor([gaussian.mean for gaussian in self._gaussians])

    @property
    def covariances(self):
        """Return the covariances (shape: KxDxD): the covariance of each Gaussian"""
        return torch.tensor([gaussian.covariance for gaussian in self._gaussians])

    @property
    def precisions(self):
        """Return the precisions (shape: KxDxD): the precision of each Gaussian"""
        return torch.tensor([gaussian.precision for gaussian in self._gaussians])

    @property
    def mean(self):
        r"""Return the expected value (i.e. mean) of the GMM: :math:`\mu = \sum_{k=1}^K \pi_k \mu_k`"""
        return torch.sum(self.priors * self.means.t(), dim=1)

    @property
    def covariance(self):
        r"""Return the covariance of the GMM: :math:`cov = \sum_{k=1}^K \pi_k (\Sigma_k + \mu_k\mu_k^T) - \mu\mu^T`"""
        cov = torch.sum([prior * (g.covariance + torch.ger(g.mean, g.mean))
                      for prior, g in zip(self.priors, self.gaussians)], dim=0)
        return cov - torch.ger(self.mean, self.mean)

    @property
    def precision(self):
        r"""Return the precision of the GMM: :math:`\Lambda = \Sigma^{-1}` (see `covariance` property)"""
        return torch.inverse(self.covariance)

    @property
    def gaussian(self):
        """Perform moment matching to approximate a GMM as a Gaussian"""
        return Gaussian(self.mean, self.covariance)

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other):
        """Copy the given GMM"""
        if not isinstance(other, GMM):
            raise ValueError("Expecting to copy another GMM")
        pass

    @staticmethod
    def is_parametric():
        r"""
        The GMM is a semi-parametric model, where the parametric part is due to the :math:`\pi_k`, and the
        non-parametric part is due to the Gaussian distributions with mean and covariance :math:`\mu_k` and
        :math:`\Sigma_k`.

        Returns:
            True
        """
        # TODO: hum... GMM are semi-parametric models...
        return True

    @staticmethod
    def is_linear():
        r"""
        The parameters of the GMM are :math:`\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K`, where the GMM is linear
        only with respect to the :math:`\pi_k`.
        """
        # TODO: not clear
        return True

    @staticmethod
    def is_recurrent():
        """
        A GMM is not a recurrent model, that is, it is not a model that 'remembers' previous inputs.
        """
        return False

    @staticmethod
    def is_probabilistic():
        """
        A GMM is a probabilistic model.
        """
        return True

    @staticmethod
    def is_discriminative():
        r"""
        The GMM is a generative model which encodes the joint distribution :math:`p(x,y)` between the input :math:`x`
        and output :math:`y`. A discriminative model can be obtained by conditioning one of the variable by the other
        one :math:`p(y|x)` or :math:`p(x|y)`.
        """
        return False

    @staticmethod
    def is_generative():
        r"""
        The GMM is a generative model which encodes the joint distribution :math:`p(x,y)` between the input :math:`x`
        and output :math:`y`. Because it is generative, we can sample from it.
        """
        return True

    ###########
    # Methods #
    ###########

    def _check_initialized(self):
        """Check if the GMM has been initialized"""
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.gaussians is None:
            raise ValueError("The Gaussian distributions have not been initialized")

    def save(self, filename):
        """
        Save the model in memory.

        Args:
            filename (str): file to save the model in.
        """
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        """
        Load a model from memory.

        Args:
            filename (str): file that contains the model.
        """
        return pickle.load(filename)

    def parameters(self):
        r"""
        Return an iterator over the model parameters, which in GMM are the priors :math:`\pi_k`, the means
        :math:`\mu_k` and the covariance matrices :math:`\Sigma_k`, :math:`\forall k \in {1,...,K}`.
        """
        yield self.priors, self.means, self.covariances

    def named_parameters(self):
        r"""
        Return an iterator over the model parameters, yielding both the name and the parameter itself. In the case
        of a GMM, the parameters are the priors :math:`\pi_k`, the means :math:`\mu_k` and the covariance matrices
        :math:`\Sigma_k`, :math:`\forall k \in {1,...,K}`.
        """
        yield "priors", self.priors
        yield "means", self.means
        yield "covariances", self.covariances

    def prob(self, value):
        r"""
        Compute the probability/likelihood of the GMM evaluated at `value`.

        .. math:: p(X | \theta) = \prod_{n=1}^N \sum_{k=1}^K \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)

        where :math:`X \in \mathbb{R}^{N \times D}` is the data matrix, and :math:`\theta = {\pi_k, \mu_k, \Sigma_k}`
        are the parameters of the GMM.

        Args:
            value (torch.Tensor[N,D]): data vector/matrix to evaluate the likelihood.

        Returns:
            float: likelihood
        """
        return torch.exp(self.log_prob(value))

    # alias
    likelihood = prob
    pdf = prob

    def log_prob(self, value):  # score() in sklearn
        r"""
        Returns the log of the probability density/mass function (i.e. the log-likelihood of the GMM) evaluated at
        `value`.

        .. math:: \log p(X | \theta) = \sum_{n=1}^N \log \sum_{k=1}^K \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)

        where :math:`X \in \mathbb{R}^{N \times D}` is the data matrix, and :math:`\theta = {\pi_k, \mu_k, \Sigma_k}`
        are the parameters of the GMM.

        Args:
            value (torch.Tensor[N,D]): data vector/matrix to evaluate the likelihood.

        Returns:
            float: log-likelihood
        """
        gaussians = torch.tensor([gaussian(x) for gaussian in self.gaussians]).t()
        return torch.sum(self.priors * gaussians)

    # alias
    log_likelihood = log_prob
    log_pdf = log_prob

    def joint_pdf(self, x, z):
        r"""
        Compute the joint probability distribution :math:`p(X,Z)`. This is the same as the complete-data likelihood
        of the GMM.

        .. math::

            P(X,Z | \theta) &= \prod_{n=1}^N p(x_n, z_n | \theta) \\
                            &= \prod_{n=1}^N p(z_n) p(x_n | z_n, \theta) \\
                            &= \prod_{n=1}^N \prod_{k=1}^K \pi_k^{z_{nk}} \mathcal{N}(x_n | \mu_k, \Sigma_k)^{z_{nk}}

        where :math:`X \in \mathbb{R}^{N \times D}` is the data matrix, :math:`Z \in \mathbb{R}^{N \times K}` is the
        associated hidden variable matrix where each entry is a binary value :math:`\{0,1\}` and each row sums up to 1,
        and :math:`\theta = {\pi_k, \mu_k, \Sigma_k}` are the parameters of the GMM.

        Args:
            x (torch.Tensor[D], torch.Tensor[N,D]): data vector/matrix
            z (int, torch.int[N], torch.int[N,K]): hidden variable index / indices, or hidden variable matrix where each
                row is a one hot encoding vector (i.e. all the elements of the row are 0 except one which has a value
                of 1)

        Returns:
            float: joint probability distribution (aka complete-data likelihood)
        """
        # if hidden variable is an index
        if isinstance(z, int):
            joints = self.priors[z] * self.gaussians[z].pdf(x)  # shape: 1 if data vector, or N if data matrix
            return torch.prod(joints)

        if isinstance(z, torch.Tensor) and len(z.shape) <= 2:
            # quick check about dimensions
            Nz = 1 if len(z.shape) == 1 else z.shape[0]
            Nx = 1 if len(x.shape) == 1 else x.shape[0]
            if Nx != Nz:
                raise ValueError("The number of samples between the data and the hidden variables should be "
                                 "the same. Got instead {} and {} respectively".format(Nx, Nz))

            # get hidden variable indices
            z_idx = z.argmax(axis=1) if len(z.shape) == 2 else z

            # compute individual joint distribution
            likelihoods = torch.tensor([g.pdf(x) for g in self.gaussians]).t()  # shape: K if vector, or NxK if matrix
            priors = torch.tensor([self.priors[z_id] for z_id in z_idx])  # shape: 1 if vector, or N if matrix
            joints = priors * likelihoods[range(Nx), z_idx]     # shape: N

            # return product of joint distributions
            return torch.prod(joints)
        else:
            raise TypeError("The given z should be an integer representing the hidden variable index, or an "
                            "array of N integers representing the hidden variable indices, or a matrix of shape NxK "
                            "where each row is a one hot encoding vector")

    # alias
    complete_data_likelihood = joint_pdf

    def log_joint_pdf(self, x, z):
        r"""
        Compute the log joint probability distribution :math:`\log p(X,Z)`. This is the same as the complete-data
        log-likelihood of the GMM.

        This is given by:

        .. math::

            \log p(X,Z|\theta) = \sum{n=1}^N \sum{k=1}^K z_{nk} (\log \pi_k + \log \mathcal{N}(x_n | \mu_k, \Sigma_k))

        where :math:`X \in \mathbb{R}^{N \times D}` is the data matrix, :math:`Z \in \mathbb{R}^{N \times K}` is the
        associated hidden variable matrix where each entry is a binary value :math:`\{0,1\}` and each row sums up to 1,
        and :math:`\theta = {\pi_k, \mu_k, \Sigma_k}` are the parameters of the GMM.

        Args:
            x (torch.Tensor): data vector/matrix to evaluate the complete-data log-likelihood.
            z (int, torch.int[N], torch.int[N,K]): hidden variable index / indices, or hidden variable matrix where each
                row is a one hot encoding vector (i.e. all the elements of the row are 0 except one which has a value
                of 1)

        Returns:
            float: log joint probability distribution (aka complete-data log-likelihood)
        """
        return torch.log(self.joint_pdf(x, z))

    # alias
    complete_data_log_likelihood = log_joint_pdf

    def posterior_pdf(self, x, z):
        r"""
        Evaluate the posterior distribution on the hidden variables :math:`Z`. This one also factorizes with respect
        to the number of data points [1], and is given by:

        .. math:: p(Z | X, \theta) = \prod_{n=1}^N p(z_n | x_n, \theta)

        where :math:`\theta` are the parameters of the GMM.

        Note that compared to the `responsibilities` method, this returns the complete posterior (i.e. a float number).

        Args:
            x (torch.Tensor): data vector/matrix
            z (int, torch.int[N], torch.int[N,K]): hidden variable index / indices, or hidden variable matrix where each
                row is a one hot encoding vector (i.e. all the elements of the row are 0 except one which has a value
                of 1)

        Returns:
            float: posterior

        References:
            [1] "Pattern Recognition and Machine Learning" (eq. 9.75), Bishop, 2006
        """
        # if hidden variable is an index
        if isinstance(z, int):
            return torch.prod(self.responsibilities(x, k=z))

        if isinstance(z, torch.Tensor) and len(z.shape) <= 2:
            # quick check about dimensions
            Nz = 1 if len(z.shape) == 1 else z.shape[0]
            Nx = 1 if len(x.shape) == 1 else x.shape[0]
            if Nx != Nz:
                raise ValueError("The number of samples between the data and the hidden variables should be "
                                 "the same. Got instead {} and {} respectively".format(Nx, Nz))

            # get hidden variable indices
            z_idx = z.argmax(dim=1) if len(z.shape) == 2 else z

            # compute individual posteriors
            posteriors = self.responsibilities(x, z_idx, axis=0)

            # return product of posteriors
            return torch.prod(posteriors)
        else:
            raise TypeError("The given z should be an integer representing the hidden variable index, or an "
                            "array of N integers representing the hidden variable indices, or a matrix of shape NxK "
                            "where each row is a one hot encoding vector")

    def log_posterior_pdf(self, x, z):
        r"""
        Evaluate the log posterior distribution on the hidden variables :math:`Z`. This one also factorizes with
        respect to the number of data points [1], and is given by:

        .. math:: \log p(Z | X, \theta) = \sum_{n=1}^N \log p(z_n | x_n, \theta)

        where :math:`\theta` are the parameters of the GMM.

        Note that compared to the `responsibilities` method, this returns the complete posterior (i.e. a float number).

        Args:
            x (torch.Tensor): data vector/matrix
            z (int, torch.int[N], torch.int[N,K]): hidden variable index / indices, or hidden variable matrix where each
                row is a one hot encoding vector (i.e. all the elements of the row are 0 except one which has a value
                of 1)

        Returns:
            float: log posterior

        References:
            [1] "Pattern Recognition and Machine Learning" (eq. 9.75), Bishop, 2006
        """
        return torch.log(self.posterior_pdf(x, z))

    def expected_complete_data_log_likelihood(self, x):
        r"""
        Expectation of the complete data log-likelihood under the posterior distribution of the latent variables.
        This is the quantity that is being maximized during the M step of the EM algorithm.

        .. math::

            Q(\theta, \theta_{old}) &= \sum_{Z} p(Z | X, \theta_{old}) \log p(X,Z | \theta) \\
                &= \sum_{n=1}^N \sum_{k=1}^K \r_k(x_n) (\log \pi_k + \log \mathcal{N}(x_n | \mu_k, \Sigma_k))

        This is linked to the lower bound :math:`\mathcal{L}(q, \theta)` where when :math:`q(Z) = p(Z|X,\theta_{old})`,
        we have: :math:`\mathcal{L}(q, \theta) = Q(\theta, \theta_{old}) + H(q)`. We can thus see that maximizing
        the expectation of the complete data log-likelihood wrt the parameters :math:`\theta` is the same as
        maximizing the lower bound wrt :math:`\theta` while holding :math:`q(Z)` fixed, which is performed during
        the M step of the EM algorithm. [1]

        Note that the summation is no longer inside the logarithm as it was the case for :math:`\log p(X|\theta)`,
        and closed-form solutions can be obtained that maximizes this loss :math:`Q(\theta, \theta_{old})` with
        respect to the parameters :math:`\theta`.

        In the GMM case, this results in:

        .. math::

            \mu_k &= \frac{1}{N_k} \sum_{n=1}^N r_k(x_n) x_n \\
            \Sigma_k &= \frac{1}{N_k} \sum_{n=1}^N r_k(x_n) (x_n - \mu_k)(x_n - \mu_k)^T \\
            \pi_k &= \frac{N_k}{N}

        where :math:`N_k = \sum_{n=1}^N r_k(x_n)`, and :math:`r_k(x_n) = p(z_k = 1|x_n)` are the responsibilities.

        Args:
            x (torch.Tensor[N,D], torch.Tensor[D]): data vector/matrix

        References:
            [1] "Pattern Recognition and Machine Learning" (chap 9.4), Bishop, 2006

        Returns:
            float: expected value of the complete data log-likelihood (under the posterior distribution of the latent
                variables)
        """
        # get useful variables
        responsibilities = self.responsibilities(x)     # shape: K if one data point, otherwise NxK
        likelihoods = torch.tensor([g.pdf(x) for g in self.gaussians]).t()  # shape: K if one data point, otherwise NxK
        priors = self.priors    # shape: K

        # compute each term of the expectation of the complete data log-likelihood under the posterior distribution
        # of the latent variables.
        q = responsibilities * (torch.log(priors) + torch.log(likelihoods))   # shape: K if data point, else NxK

        # sum over all latent variables and data points
        return torch.sum(q)

    def aic(self, x):
        r"""
        Return the Akaike Information Criterion (AIC) for the current model on the data x. The lower the better.

        .. math:: AIC = - 2 \log(\mathcal{L}(x, \theta)) + 2 n_p

        where :math:`\mathcal{L}(x, \theta)` is the likelihood of the model, :math:`n_p` is the number of free
        parameters required for a GMM of :math:`K` components, i.e. :math:`n_p = (K-1) + K(D + 1/2 D(D+1))`, :math:`N`
        is the number of data points, and :math:`D` is the dimensionality of the data.

        Args:
            x (torch.Tensor): data vector/matrix.

        Returns:
            float: AIC score
        """
        return - 2 * self.log_likelihood(x) + 2 * self.num_parameters

    def bic(self, x):
        r"""
        Return the Bayesian Information Criterion (BIC) score which can be used to estimate the number of Gaussians.
        The lower this number is, the better.

        .. math:: BIC = - 2 \log(\mathcal{L}(x, \theta)) + n_p \log(N)

        where :math:`\mathcal{L}(x, \theta)` is the likelihood of the model, :math:`n_p` is the number of free
        parameters required for a GMM of :math:`K` components, i.e. :math:`n_p = (K-1) + K(D + 1/2 D(D+1))`, :math:`N`
        is the number of data points, and :math:`D` is the dimensionality of the data.

        Args:
            x (torch.Tensor): data vector/matrix.

        Returns:
            float: BIC score
        """
        return - 2 * self.log_likelihood(x) + self.num_parameters * torch.log(self.num_data)

    def init_random(self, data, seed=None, reg=1e-8):
        r"""
        Initialize the GMM randomly.

        Args:
            data (torch.Tensor): data matrix
            seed (int, None): seed for random generator
            reg (float): regularization term (useful to not have singular covariance matrices)
        """
        # initialize random generator
        self.seed = seed

        # uniform priors
        self._priors = torch.ones(self.num_components) / self.num_components

        # compute mean and covariance of the data
        mean = Gaussian.compute_mean(data, dim=0)
        cov = Gaussian.compute_covariance(data, axis=0, bessels_correction=True)

        # generate means from a multivariate normal
        # means = np.random.multivariate_normal(mean, cov, size=self.num_components)
        means = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov).sample((self.num_components,))

        # covariances (same as the above covariance + reg)
        cov = cov + reg * torch.eye(self.dim)
        covariances = torch.tensor([cov] * self.num_components)

        # create gaussians
        self._gaussians = [Gaussian(mean=mu, covariance=cov) for mu, cov in zip(means, covariances)]

    def init_linear(self, data):
        r"""
        Initialize the GMM uniformly and linearly in the space. This initialization scheme is deterministic.

        Args:
            data (torch.Tensor[N,D]): data matrix
        """
        # compute lower and upper bounds
        lower_bound, upper_bound = torch.min(data, dim=0), torch.max(data, dim=0)

        # distribute the means
        means = []

        # compute the covariances
        covariances = []

        # create gaussians
        self._gaussians = [Gaussian(mean=mu, covariance=cov) for mu, cov in zip(means, covariances)]

    def init_uniformly(self, data, axis=0):
        r"""
        Initialize the GMM uniformly in the space with respect to the axis dimension. If the data represents
        trajectories, the first dimension is the time and it will distributed uniformly with respect to that one.

        Args:
            data (torch.Tensor[N,D]): data matrix
            axis (int): axis specifying the dimension
        """
        # compute lower and upper bounds
        lower_bound, upper_bound = torch.min(data, dim=0), torch.max(data, dim=0)

        # distribute linearly with respect to the specified axis/dimension
        distance = upper_bound[axis] - lower_bound[axis]
        x = distance / (self.num_components + 1.) * torch.arange(self.num_components)

        # compute centers for the other dimensions
        idx = torch.arange(len(x)) != axis
        centers = (upper_bound[idx] - lower_bound[idx]) / 2.

        # compute means (combine the centers with the distribution over the specified axis)
        means = []

        # compute covariances
        covariances = []

        # create gaussians
        self._gaussians = [Gaussian(mean=mu, covariance=cov) for mu, cov in zip(means, covariances)]

    def init_curvature(self, data):
        r"""
        Initialize the GMM using the curvature of the trajectories. This only works for sequential spatial data.

        Args:
            data (torch.Tensor[T,D], torch.Tensor[N,T,D]): data matrix or vector
        """
        pass

    def init_sklearn(self, data):
        r"""
        Initialize the GMM using the sklearn library.

        Args:
            data (torch.Tensor[T,D], torch.Tensor[N,T,D]): data matrix or vector
        """
        pass

    def init_time_warping(self, data):
        r"""
        Initialize the GMM using Dynamic Time Wrapping [1]. This only works for sequential temporal and spatial data.

        Args:
            data:

        References:
            [1] "Robot Programming by Demonstration: A Probabilistic Approach", Calinon, 2009, Chap 2.9.3
        """
        pass

    def init_kmeans(self, data, seed=None):
        r"""
        Initialize the GMM using K-means algorithm.

        Args:
            data (torch.Tensor[N,D]): data matrix
            seed (int, None): seed for random generator
        """
        # initialize random generator
        self.seed = seed

        # fit the data using k-means
        km = KMeans(n_clusters=self.num_components)
        km.fit(data)

        # uniform priors
        self._priors = torch.ones(self.num_components) / self.num_components

        # identity covariances
        covariances = torch.tensor([torch.eye(self.dim)] * self.num_components)

        # means = position of the cluster centers
        means = torch.from_numpy(km.cluster_centers_).float()

        # create gaussians
        self._gaussians = [Gaussian(mean=mu, covariance=cov) for mu, cov in zip(means, covariances)]

    def init(self, data, method='k-means', seed=None, reg=1e-8):
        r"""
        Initialize the GMM using the specified method

        Args:
            data (torch.Tensor[N,D]): data matrix
            method (str, None): 'k-means', 'random', None. If None, it starts from where the Gaussians are placed.
            seed (str): seed for random generator
            reg (float): regularization term (useful to not have singular covariance matrices)
        """
        if method is None:
            return
        method = method.lower()
        if method == 'random':
            self.init_random(data, seed, reg=reg)
        elif method == 'k-means' or method == 'kmeans':
            self.init_kmeans(data, seed)
        else:
            raise NotImplementedError("The given initialization method has not been implemented")

    def expectation_maximization(self, X, reg=1e-8, num_iters=1000, threshold=1e-4,
                                 init='kmeans', seed=None, verbose=False):
        r"""
        Fit the GMM to the provided data by performing the expectation maximization algorithm. The EM algorithm allows
        to find the maximum likelihood estimate for models that have latent variables. This one consists of 4 main
        steps:
        1. Initialize the parameters :math:`\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K`
        2. Expectation step: evaluate the posterior :math:`p(Z | X, \theta_{old})` while fixing the parameters.
        3. Maximization step: maximize the expected value of the complete-data log-likelihood under the posterior
        distribution of the latent variables (found during the Expectation step). That is,
        :math:`\max_\theta Q(\theta, \theta_{old}) = \max_\theta \sum_{Z} p(Z | X, \theta_{old}) \log p(X,Z | \theta)`.
        4. Evaluate the log-likelihood loss, and check if it converged. If it didn't, go back to step 2.

        To be more specific, the EM algorithm alternatively computes a lower bound on the log-likelihood for the
        current parameters, and then maximize this bound to obtain the new parameter values (see [1], sec 9.4 for
        more details). Note that the lower bound and the expected value of the complete-data log-likelihood under
        the posterior distribution of the latent variables are related.

        In the GMM case, the E-step consists to compute the responsibilities: :math:`r_k(x_n)`, and the M-step
        consists to compute the following equations (which are the closed-form solutions to the maximization of
        the lower bound):

        .. math::

            \mu_k &= \frac{1}{N_k} \sum_{n=1}^N r_k(x_n) x_n \\
            \Sigma_k &= \frac{1}{N_k} \sum_{n=1}^N r_k(x_n) (x_n - \mu_k)(x_n - \mu_k)^T \\
            \pi_k &= \frac{N_k}{N}

        where :math:`N_k = \sum_{n=1}^N r_k(x_n)`, and :math:`r_k(x_n) = p(z_k = 1|x_n)` are the responsibilities.

        Args:
            X (torch.Tensor[N,D]): data matrix
            reg (float): regularization term
            num_iters (int): number of iterations
            threshold (float): convergence threshold
            init (str, None): how the Gaussians should be initialized. Possible values are 'random', 'kmeans', and
                None. If None, it will use the initial positions of the provided gaussians.
            seed (int, None): seed for random generator
            verbose (bool): if we should print details during the optimization process

        Returns:
            dict: dictionary containing info collected during the optimization process, such as the history of losses,
                the number of iterations it took to converge, if it succeeded, etc.
        """
        # quick check
        if len(X.shape) != 2:
            raise ValueError("Expecting a 2D array of shape NxD for the data")

        # compute dictionary results
        results = {'losses': [], 'success': False, 'num_iters': 0}
        self.N = X.shape[0]

        # 1. Initialize
        self.init(X, method=init, seed=seed, reg=reg)
        if init is None:
            self.seed = seed

        # compute initial loss
        loss = self.log_likelihood(X)
        prev_loss = loss
        results['losses'].append(loss)

        for it in range(num_iters):
            # 2. E-step
            r_kn = self.responsibilities(X)      # shape: NxK

            # 3. M-step
            N_k = torch.sum(r_kn, dim=0)        # shape: K
            mu_k = (1./N_k * X.t().matmul(r_kn)).t()     # shape: KxD
            diff = X - mu_k[k]
            cov_k = torch.tensor([(r_kn[:, k] * diff.t()).matmul(diff) for k in range(self.K)])  # KxDxD
            cov_k = (1./N_k * cov_k.t()).t()  # KxDxD
            self._priors = N_k / self.N        # shape: K
            self._gaussians = [Gaussian(mean=mu, covariance=cov) for mu, cov in zip(mu_k, cov_k)]

            # 4. check convergence
            loss = self.log_likelihood(X)
            results['losses'].append(loss)
            if torch.abs(loss - prev_loss) <= threshold:
                if verbose:
                    print("Convergence achieved at iteration {} with associated loss: {}".format(it+1, loss))
                results['num_iters'] = it+1
                results['success'] = True
                return results

            # update previous loss
            prev_loss = loss

        return results

    # aliases
    em = expectation_maximization
    fit = expectation_maximization
    mle = expectation_maximization
    maximum_likelihood = expectation_maximization

    def predict(self, x):
        r"""
        Predict from which component the data is from, and return the index of this component/Gaussian.

        Args:
            x (torch.Tensor): data vector/matrix

        Returns:
            int, torch.Tensor: component index/indices
        """
        posteriors = self.responsibilities(x)  # shape NxK if data matrix, or K if data vector
        if len(posteriors.shape) == 2:
            idx = torch.argmax(posteriors, dim=1)
        else:
            idx = torch.argmax(posteriors)
        return idx

    def predict_prob(self, x):
        r"""
        Predict from which component the data is from with the associated probability.

        Args:
            x (torch.Tensor): data vector/matrix

        Returns:
            int, torch.int[N]: component index/indices
            float, torch.float[N]: associated probability
        """
        posteriors = self.responsibilities(x) # shape NxK if data matrix, or K if data vector
        if len(posteriors.shape) == 2:
            idx = torch.argmax(posteriors, dim=1)
        else:
            idx = torch.argmax(posteriors)
        return idx, posteriors[idx]

    def cumulative_prior(self):
        r"""
        Return the cumulative distribution function on the prior :math:`\sum_{k=1}^{m} \pi_k \forall m \in {1, ..., K}`.
        Note that the order in the sum is important here.

        Returns:
            torch.float[K]: cumulative distribution function on the prior
        """
        return torch.cumsum(self.priors)

    def sample_hidden(self, size=None, seed=None):
        r"""
        Sample from hidden random variable :math:`Z` (i.e. from the priors).

        Args:
            size (int, None): number of samples
            seed (int, None): seed for the random number generator

        Returns:
            int, torch.int[N]: component indices
        """
        self.seed = seed

        # sample from uniform distribution
        if size is None:
            size = 1
        random = torch.rand(size, requires_grad=True)

        # compute cumulative distribution function on priors
        cumsum = self.cumulative_prior()
        prior_idx = torch.range(start=0, end=len(cumsum), requires_grad=True)

        # sample the components
        idx = torch.tensor([prior_idx[cumsum < rand_number][-1] for rand_number in random], requires_grad=True)

        # if one sample, just return this one
        if idx.size == 1:
            return idx[0]

        # otherwise, return all of them
        return idx

    def sample(self, size=None, seed=None, kind=None):
        r"""
        Generate `size` samples from the GMM. This uses the ancestral/forward sampling method; that is, it first
        samples from the hidden variables :math:`\hat{z} \sim p(z)`, and then from the conditional distribution
        :math:`\hat{x} \sim p(x|\hat{z})`.

        Args:
            size (int, None): number of samples
            seed (int, None): seed for the random number generator
            kind (str): if kind == 'complete', it returns the data along with from which component (i.e. Gaussian)
                it was sampled from. Otherwise, it just returns the data.

        Return:
            torch.Tensor[D], torch.Tensor[N,D]: samples
            int, torch.int[N]: component index/indices (if the argument kind == 'complete')
        """
        idx = self.sample_hidden(size=size, seed=seed)
        if isinstance(idx, int):  # just one
            return self.gaussians[idx].rsample()  # shape: D
        return torch.tensor([self.gaussians[i].rsample() for i in idx])  # shape: N, D

    def responsibilities(self, x, k=None, axis=1):
        r"""
        Compute the responsibilities (posterior probability of component k once we have observed the data `x`).
        These are given by:

        .. math::

            r_k(x) &= p(z_k=1 | x) \\
                   &= \frac{p(x, z_k=1)}{p(x)} \\
                   &= \frac{p(z_k=1) p(x | z_k=1)}{\sum_{j=1}^{K} \p(z_j=1) p(x | z_j=1)}
                   &= \frac{\pi_k \mathcal{N}(x|\mu_k,\Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x|\mu_j,\Sigma_j)}

        Args:
            x (torch.Tensor): data vector/matrix
            k (torch.Tensor, int, slice, None): component index(ices). If None, compute the responsibilities wrt to each
                component.
            axis (int): This argument is useful when the argument 'k' is an array; k can then be an array of size `N`
                or between 1 and `K`. If `N` is bigger than `K` then we can infer what the user wants, however if `N`
                is smaller than `K`, then we have to specify what the nature of `k` is; Does the user wants back a
                `Nxk` matrix or `Nx1` vector? That is, the indices in k are the ones that we are interested to get
                back, or they are indices for each sample? Because the number of data points is often bigger than `K`,
                this argument is not used, but in the case `N` is smaller than `K`, then by default axis == 1, which
                means that it will return a `Nxk` matrix. If axis == 0, it will return an array of size `N` which
                contains the responsibilities :math:`h_{k_n}(x_n)`.

        Returns:
            float, torch.Tensor: responsibility(ies). It will be a float number if only one datapoint and one component
                index were provided. An array of size `k` if one data point and `k` component indices were provided.
                An array of size `N` if multiple data points and one component index or `N` component indices were
                given. A matrix of shape `Nxk` if multiple data points and `k` component indices were given.
        """
        gaussian_pdfs = torch.tensor([g.pdf(x) for g in self.gaussians]).t()  # shape: K if 1 data point, or NxK
        # if multiple
        joint = self.priors * gaussian_pdfs  # shape: K if 1 data point, or NxK if multiple
        marginal = torch.sum(joint, dim=1)  # shape: 1 if 1 data point, or N if multiple

        if k is None:
            return (joint.t() / marginal).t()  # shape: K if 1 data point, or NxK if multiple
        else:
            N = 1 if len(x.shape) == 1 else x.shape[0]
            if N == 1:  # one data point
                return joint[k] / marginal  # shape: k
            else:  # multiple data points
                K = self.num_components
                if N == len(k):
                    if N <= K and axis == 1:
                        return (joint[:, k].t() / marginal).t()  # shape: Nxk
                    return joint[range(N), k] / marginal  # shape: N
                return (joint[:, k].t() / marginal).t()  # shape: Nxk

    def condition(self, x_in, idx_out, idx_in=None):
        r"""
        Condition the GMM which results in GMR. Return the conditioned GMM. If the user wants to approximate it
        by a single gaussian, he/she can call the property `gaussian` or the `approximate_by_single_gaussian()`
        methods. These two's are equivalent.

        Gaussian Mixture Regression [1,2] consists to condition the GMM (that models the joint distribution over the
        input and output variables :math:`p(x^I, x^O)`) on a part of the variables (for instance, the input variables
        :math:`p(x^O | x^I`). Let's :math:`x = [x^I, x^O]`, :math:`\mu_k = [\mu_k^I, \mu_k^O]`, and :math:`\Sigma_k =
        \left[ \begin{array}{cc} \Sigma_k^I & \Sigma_k^{IO} \\ \Sigma_k^{OI} & \Sigma_k^O \end{array} \right]`, where
        :math:`I` and :math:`O` are the superscripts to refer the input and output respectively.

        .. math::

            p(x^O | x^I) &= \sum_{k=1}^K p(z_k=1 | x^I) p(x^O | x^I, z_k=1) \\
                         &= \sum_{k=1}^K r_k(x^I) \mathcal{N}(\hat{\mu}_k^O(x^I), \hat{\Sigma}_k^O)

        where :math:`r_k(x^I) = \frac{\pi_k \mathcal{N}(x^I|\mu_k^I, \Sigma_k^I)}{\sum_{j=1}^{K} \pi_j
        \mathcal{N}(x^I|\mu_j^I,\Sigma_j^I)}` are the responsibilities, :math:`\hat{\mu}_k^O(x^I) =
        \mu_k^O + \Sigma_k^{OI} \Sigma_k^I^{-1} (x^I - \mu_k^I)` and :math:`\hat{\Sigma}_k^O = \Sigma_k^O -
        \Sigma_k^{OI} (\Sigma_k^I)^{-1} \Sigma_k^{IO}` are the resulting conditioned means and covariances,
        respectively.

        Args:
            x_in (float[d2]): array of values :math:`x^I` such that we have :math:`p(x^O|x^I)`
            idx_out (int[d1]): indices that we are interested in (indices of :math:`x^O` in :math:`x`) given
                (i.e. conditioned on) the other ones
            idx_in (int[d2]): indices that we conditioned on corresponding to the values. If None, it will be inferred.

        Returns:
            GMM: conditioned gaussian mixture model

        References:
            [1] "Robot Programming by Demonstration: a Probabilistic Approach" (chap 2), Calinon, 2009
            [2] "A Tutorial on Task-Parameterized Movement Learning and Retrieval", Calinon, 2015
        """
        priors = self.responsibilities(x_in)
        gaussians = [g.condition(x_in, idx_out, idx_in) for g in self.gaussians]
        return GMM(priors=priors, gaussians=gaussians)

    def marginalize(self, idx):
        r"""
        Compute and return the marginal distribution (which is also a GMM) of the specified indices.

        Let's assume that the joint distribution :math:`p(x_1, x_2)` is modeled as a GMM, that is:

        .. math:: x \sim \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)

        where :math:`x = [x_1, x_2]`, :math:`\mu = [\mu_1^{(k)}, \mu_2^{(k)}]` and
        :math:`\Sigma=\left[\begin{array}{cc} \Sigma_{11}^{(k)} & \Sigma_{12}^{(k)} \\ \Sigma_{21}^{(k)} &
        \Sigma_{22}^{(k)} \end{array}\right]`

        then the marginal distribution :math:`p(x_1) = \int_{x_2} p(x_1, x_2) dx_2` is also a GMM and is given by:

        .. math:: p(x_1) = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_1^{(k)}, \Sigma_{11}^{(k)})

        Args:
            idx (int, slice): indices of :math:`x_1` (this value should be between 0 and D-1, where D is
                the dimensionality of the data)

        Returns:
            GMM: marginal distribution (which is also a GMM)
        """
        gaussians = [gaussian.marginalize(idx) for gaussian in self.gaussians]
        return GMM(priors=self.priors, gaussians=gaussians)

    def multiply(self, other):
        r"""
        Multiply a GMM by another Gaussian or GMM, by a square matrix (under an affine transformation), or a float
        number.

        1. The product of a GMM by a Gaussian is given by:

        .. math::

            \mathcal{N}(\mu, \Sigma) \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)
            &= \sum_{k=1}^K \pi_k \mathcal{N}(\mu, \Sigma) \mathcal{N}(\mu_k, \Sigma_k) \\
            &= \sum_{k=1}^K \pi_k c_k \mathcal{N}(\hat{\mu}_k, \hat{\Sigma}_k)

        where :math:`c_k = \mathcal{N}(\mu; \mu_k, \Sigma + \Sigma_k)` are constant (scalar) coefficients,
        :math:`\hat{\Sigma}_k = (\Sigma^{-1} + \Sigma_k^{-1})^-1`, and
        :math:`\hat{\mu}_k = \hat{\Sigma}_k (\Sigma^{-1} \mu + \Sigma_k^{-1} \mu_k)`. In order for this result to
        be a proper probability distribution, we have to normalize it, which gives:

        .. math:: p(x) = \sum_{k=1}^K \hat{\pi}_k \mathcal{N}(\hat{\mu}_k, \hat{\Sigma}_k)

        where :math:`\hat{\pi}_k = \frac{c_k \pi_k}{\sum_{j=1}^K c_j \pi_j}`.


        2. Similarly, the product of two GMMs is given by:

        .. math::

            \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k) \sum_{j=1}^J \pi_j \mathcal{N}(\mu_j, \Sigma_j)
            &= \sum_{k=1}^K \sum_{j=1}^J \pi_k \pi_j \mathcal{N}(\mu_k, \Sigma_k) \mathcal{N}(\mu_j, \Sigma_j) \\
            &= \sum_{k=1}^K \sum_{j=1}^J c_{kj} \pi_k \pi_j \mathcal{N}(\mu_{kj}, \Sigma_{kj})

        where where :math:`c_{kj} = \mathcal{N}(\mu_k; \mu_j, \Sigma_k + \Sigma_j)` are constants (scalars),
        :math:`\Sigma_{kj} = (\Sigma_k^{-1} + \Sigma_j^{-1})^-1`, and
        :math:`\mu_{kj} = \Sigma_{kj} (\Sigma_k^{-1} \mu_k + \Sigma_j^{-1} \mu_j)`. In order for this result to
        be a proper probability distribution, we have to normalize it, which gives:

        .. math:: p(x) = \sum_{k=1}^K \sum_{j=1}^J \pi_{kj} \mathcal{N}(\mu_{kj}, \Sigma_{kj})

        where :math:`\pi_{kj} = \frac{c_{kj} \pi_k \pi_j}{\sum_{m=1}^K \sum_{n=1}^J c_{mn} \pi_m \pi_n}`.

        Note that the product of two GMMs increase the number of components which is equal to :math:`K*J`. If
        the user wants to multiply two GMMs element-wise, please see the `multiply_element_wise` method.


        3. The product of a GMM with a square matrix :math:`A` gives:

        .. math:: Ax \sim \sum_{k=1}^K \pi_k \mathcal{N}(A \mu_k, A \Sigma_k A^T)


        4. The product of a GMM by a float does nothing as we have to re-normalize it to be a proper distribution.

        Args:
            other (Gaussian, GMM, torch.float[D,D], float): Gaussian, GMM, square matrix (to rotate or scale), or float

        Returns:
            GMM: resulting GMM
        """
        # if other == Gaussian
        if isinstance(other, Gaussian):
            coefficients = torch.tensor([g.get_multiplication_coefficient(other) for g in self.gaussians])  # shape: K
            normalization = torch.sum(self.priors * coefficients)
            priors = self.priors * coefficients / normalization
            gaussians = [g * other for g in self.gaussians]
            return GMM(priors=priors, gaussians=gaussians)

        # if other == GMM
        elif isinstance(other, GMM):
            coefficients, priors, gaussians = [], [], []
            for prior1, gaussian1 in zip(self.priors, self.gaussians):
                for prior2, gaussian2 in zip(other.priors, other.gaussians):
                    prior = prior1 * prior2
                    coeff = gaussian1.get_multiplication_coefficient(gaussian2)
                    gaussian = gaussian1 * gaussian2

                    priors.append(prior)
                    coefficients.append(coeff)
                    gaussians.append(gaussian)

            priors, coefficients = torch.tensor(priors), torch.tensor(coefficients)
            normalization = torch.sum(priors * coefficients)
            priors = priors * coefficients / normalization
            return GMM(priors=priors, gaussians=gaussians)

        # if other == square matrix
        elif isinstance(other, torch.Tensor):
            return self.affine_transform(other)

        # if other == number
        elif isinstance(other, (int, float)):
            return self

        else:
            raise TypeError("Trying to multiply a Gaussian with {}, which has not be defined".format(type(other)))

    def multiply_element_wise(self, other):
        r"""
        Multiply element wise two GMMs. If a Gaussian, square matrix, or float is given, it will just call
        the `multiply` method.

        Compared to the `multiply` method, the element-wise multiplication between two GMMs does not increase the
        number of components.

        Args:
            other (Gaussian, GMM, torch.float[D,D], float): Gaussian, GMM, square matrix (to rotate or scale), or float

        Returns:
            GMM: resulting GMM
        """
        if isinstance(other, GMM):
            coefficients, priors, gaussians = [], [], []
            for (prior1, gaussian1), (prior2, gaussian2) in zip(self, other):
                prior = prior1 * prior2
                coeff = gaussian1.get_multiplication_coefficient(gaussian2)
                gaussian = gaussian1 * gaussian2

                priors.append(prior)
                coefficients.append(coeff)
                gaussians.append(gaussian)

            priors, coefficients = torch.tensor(priors), torch.tensor(coefficients)
            normalization = torch.sum(priors * coefficients)
            priors = priors * coefficients / normalization
            return GMM(priors=priors, gaussians=gaussians)
        else:
            return self.multiply(other)

    def add(self, other):
        r"""
        Add a GMM with a Gaussian, or a GMM with a vector (affine transformation).

        1. The sum of a GMM with a Gaussian (which is independent) results in:

        .. math::

            \mathcal{N}(\mu, \Sigma) + \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)
            =  \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k + \mu, \Sigma_k + \Sigma)

        The sum of two independent Gaussian RVs (with the same dimensionality), such that
        :math:`x_1 \sim \mathcal{N}(\mu_1, \Sigma_1)` and :math:`x_2 \sim \mathcal{N}(\mu_2, \Sigma_2)`, is given
        by :math:`x_1 + x_2 \sim \mathcal{N}(\mu_1 + \mu_2, \Sigma_1 + \Sigma_2)`

        2. The sum of a GMM :math:`x \sim \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)` with a vector :math:`v`
        results in a translation of this distribution, given by
        :math:`x \sim  \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k + v, \Sigma_k)`.

        Args:
            other (Gaussian, torch.Tensor[D]): the other Gaussian distribution, or a vector.

        Returns:
            GMM: resulting GMM
        """
        if isinstance(other, Gaussian):
            gaussians = [Gaussian(g.mean + other.mean, g.covariance + other.covariance) for g in self.gaussians]
            return GMM(priors=self.priors, gaussians=gaussians)
        elif isinstance(other, torch.Tensor):
            gaussians = [Gaussian(g.mean + other) for g in self.gaussians]
            return GMM(priors=self.priors, gaussians=gaussians)
        else:
            raise NotImplementedError("Addition not defined for the given type {}".format(type(other)))

    def affine_transform(self, A, b=None):
        r"""
        Perform an affine transformation on the GMM. For a GMM, we have
        :math:`x \sim \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)`, then
        :math:`Ax+b \sim \sum_{k=1}^K \pi_k \mathcal{N}(A \mu_k + b, A \Sigma_k A^T)`.

        Args:
            A (torch.Tensor[D,D]): square matrix
            b (torch.Tensor[D]): vector

        Returns:
            GMM: resulting GMM
        """
        gaussians = [g.affine_transform(A, b) for g in self.gaussians]
        return GMM(priors=self.priors, gaussians=gaussians)

    def integrate(self, lower=None, upper=None):
        r"""
        Integrate the GMM between the two given bounds.

        .. math::

            \int p(x) dx &= \int_{x_0}^{x_f} \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k) dx \\
                         &= \sum_{k=1}^K \pi_k \int_{x_0}^{x_f} \mathcal{N}(\mu_k, \Sigma_k) dx \\
                         &= \sum_{k=1}^K \pi_k \p_k(x_0 <= x <= x_f)

        Args:
            lower (torch.Tensor[D], float, None): lower bound (default: -np.inf)
            upper (torch.Tensor[D], float, None): upper bound (default: np.inf)

        Returns:
            float: p(lower <= x <= upper)
        """
        probs = torch.tensor([g.integrate(lower, upper) for g in self.gaussians])
        return torch.sum(self.priors * probs)

    def grad(self, x, k=None, wrt='x'):
        r"""
        Compute the gradient of the GMM evaluated at the given data 'x' with respect to the specified variable and
        component 'k'. Let's :math:`p(x; {\pi_k, \mu_k, \Sigma_k}_{k=1}^K) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k,
        \Sigma_k)` be the GMM. Then (using [1]), we have:

        .. math::

            \frac{\partial p(x)}{\partial x} &= - \sum_{k=1}^K \pi_k \mathcal{N}_k \Lambda_k (x - \mu_k) \\
            \frac{\partial p(x)}{\partial \pi_j} &= \mathcal{N}_j \\
            \frac{\partial p(x)}{\partial \mu_j} &= \pi_j \mathcal{N}_j \Lambda_j (x - \mu_j) \\
            \frac{\partial p(x)}{\partial \Sigma_j} &= \frac{\pi_j}{2} \mathcal{N}_j (\Lambda_j (x-\mu_j)(x-\mu_j)^T
                \Lambda_j - \Lambda_j) \\
            \frac{\partial p(x)}{\partial \Lambda_j} &= \frac{\pi_j}{2} \mathcal{N}_j (\Sigma_j - (x-\mu_j)(x-\mu_j)^T)

        where :math:`\Lambda = \Sigma^{-1}` is the precision matrix, and
        :math:`\mathcal{N}_j = \mathcal{N}(\mu_j, \Sigma_j)` is a multivariate Gaussian distribution.

        Args:
            x (torch.Tensor[D]): data vector
            k (int, None): index of the component. If None, it will return the gradient for each component if 'wrt'
                is different from 'x'.
            wrt (str): specify with respect to which variable we compute the gradient. It can take the following
                values 'x', 'pi' or 'prior', 'mu' or 'mean', 'sigma' or 'covariance', 'lambda' or 'precision'.

        Returns:
            torch.Tensor: gradient of the same shape (as the variable from which we take the gradient)

        References:
            [1] "The Matrix Cookbook" (math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), Petersen and Pedersen, 2012
        """
        # TODO: check when x is a matrix

        # if the derivative wrt a component is specified
        if k is not None:
            if wrt == 'pi' or wrt == 'prior':
                return self.gaussians[k].pdf(x)
            return self.gaussians[k].grad(x, wrt=wrt)

        wrt = wrt.lower()
        if wrt == 'x':
            return torch.sum([prior * g.grad(x, wrt=wrt) for prior, g in zip(self.priors, self.gaussians)], dim=0)
        elif wrt == 'pi' or wrt == 'prior':
            return torch.tensor([gaussian.pdf(x) for gaussian in self.gaussians])
        elif wrt == 'mu' or wrt == 'mean' or wrt == 'sigma' or wrt[:3] == 'cov' \
                or wrt == 'lambda' or wrt == 'precision':
            return torch.tensor([prior * g.grad(x, wrt=wrt) for prior, g in zip(self.priors, self.gaussians)])
        else:
            raise ValueError("The given 'wrt' argument is not valid (see documentation)")

    def grad_log_likelihood(self, x):
        pass

    def hessian(self, x, wrt='x'):
        pass

    def update(self, x):
        r"""
        Online update of the GMM given new data points.

        Args:
            x (torch.Tensor): data vector/matrix
        """
        pass

    def approximate_by_single_gaussian(self):
        r"""
        Approximate the GMM by a single Gaussian. This is the same as the `gaussian` property. Let's the GMM be
        defined as :math:`p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)`.

        The mean is then given by :math:`\mu = \mathbb{E}_{x \sim p(x)}[x]`:

        .. math::

            \mathbb{E}_{x \sim p(x)}[x] &= \int x p(x) dx \\
                                        &= \int x \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k) dx \\
                                        &= \sum_{k=1}^K \pi_k \int x \mathcal{N}(\mu_k, \Sigma_k) dx \\
                                        &= \sum_{k=1}^K \pi_k \mu_k

        The covariance is given by :math:`\Sigma = \mathbb{E}_{x \sim p(x)}[xx^T] - \mathbb{E}_{x \sim p(x)}[x]
        \mathbb{E}_{x \sim p(x)}[x]^T`. Let's focus on :math:`\mathbb{E}_{x \sim p(x)}[xx^T]`.

        .. math::

            \mathbb{E}_{x \sim p(x)}[xx^T] &= \int xx^T p(x) dx \\
                                      &= \int xx^T \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k) dx \\
                                      &= \sum_{k=1}^K \pi_k \int xx^T \mathcal{N}(x|\mu_k, \Sigma_k) dx \\
                                      &= \sum_{k=1}^K \pi_k \mathbb{E}_{x \sim \mathcal{N}(x|\mu_k, \Sigma_k)}[xx^T]
                                      &= \sum_{k=1}^K \pi_k (\Sigma_k + \mu_k \mu_k^T)

        Thus, the covariance is given by: :math:`\Sigma(x) = \sum_{k=1}^K \pi_k (\Sigma_k + \mu_k \mu_k^T) - \mu \mu^T`

        Returns:
            Gaussian: gaussian distribution
        """
        return Gaussian(mean=self.mean, covariance=self.covariance)

    # alias
    moment_matching = approximate_by_single_gaussian

    def entropy(self):
        r"""
        Differential entropy associated with the GMM distribution.

        .. math::

            H(x) &= - \int p(x) \ln p(x) dx \\
                 &= - \sum_{k=1}^K \pi_k \int \mathcal{N}_k(x) \ln ( \sum_{k=1}^K \pi_k \mathcal{N}_k(x) ) dx

        where :math:`\mathcal{N}_k(x) = \mathcal{N}(x | \mu_k, \Sigma_k)`.

        Returns:
            float: differential entropy
        """
        pass

    def kl_divergence(self, other):
        r"""
        Compute the Kullback-Leibler divergence between the two GMMs. Note that the KL divergence between two GMMs is
        not analytically tractable, and that the KL divergence is not symmetric.

        Warnings: This only valid if the other distribution is also a GMM.

        .. math::

            D_{KL}(p_1 || p_2) = - \int p_1 \ln \left( \frac{p_2}{p_1} \right) dx

        Args:
            other (GMM): the other gaussian mixture model.

        Returns:
            float: the divergence between the 2 GMMs.
        """
        pass

    #############
    # Operators #
    #############

    def __str__(self):
        """Return name of this class"""
        return self.__class__.__name__

    def __call__(self, x=None, z=None, size=None):
        r"""
        If no arguments are provided, it returns one sample from the distribution. If the data vector or matrix is
        provided, it returns the associated probability for each sample (i.e. the likelihood that the given sample(s)
        was/were generated from this GMM), that is :math:`p(x)`. If the index `k` is also provided in addition to the
        data :math:`x`, it returns the joint probability :math:`p(x, z_k=1)`.

        Args:
            x (torch.float[N,D], torch.float[D]): data matrix/vector
            z (int, torch.int[N], None): hidden variable (component index/indices)
            size (int, None): number of samples

        Returns:
            float, or torch.Tensor: probability evaluated at `x`, or samples
        """
        if x is not None:
            return self.joint_pdf(x, z)
        return self.sample(size=size)

    def __len__(self):
        """
        Return the number of components.

        Returns:
            int: number of components
        """
        return len(self.priors)

    def __iter__(self):
        """
        Iterate over each component of the GMM.

        Returns:
            float: prior probability
            Gaussian: gaussian distribution
        """
        for prior, gaussian in zip(self.priors, self.gaussians):
            yield prior, gaussian

    def __getitem__(self, idx):
        r"""
        Return the prior probability :math:`\pi_k` and the corresponding Gaussian `N(\mu_k, \Sigma_k)` associated
        to the given index.

        Args:
            idx (int): index

        Returns:
            float: prior probability
            Gaussian: gaussian distribution
        """
        return self.priors[idx], self.gaussians[idx]

    def __add__(self, other):
        """
        Add a GMM with a Gaussian, or a GMM with a vector (affine transformation).

        Args:
            other (Gaussian, float[d]): the other Gaussian, or vector.

        Returns:
            GMM: resulting GMM
        """
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        r"""
        Multiply two GMMs, or a GMM by a Gaussian, matrix, or float. See the `multiply` method for more information.

        Warnings: the multiplication of two GMMs performed here is NOT the one that multiply the components
            element-wise. For this one, have a look at `multiply_element_wise` method, or the `__and__` operator.

        Args:
            other (GMM, Gaussian, torch.float[D,D], float): GMM, Gaussian, or square matrix (to rotate or scale), or float

        Returns:
            GMM: resulting GMM
        """
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __and__(self, other):
        r"""
        Multiply two GMMs, or a GMM by a Gaussian, matrix, or float. See the `multiply_element_wise` method for more
        information. Note that the multiplication between two GMMs performed here will multiply the components
        element-wise.

        Args:
            other (GMM, Gaussian, torch.float[D,D], float): GMM, Gaussian, or square matrix (to rotate or scale), or float

        Returns:
            GMM: resulting GMM
        """
        return self.multiply_element_wise(other)

    def __rand__(self, other):
        return self.multiply_element_wise(other)


######################
# Plotting functions #
######################

def plotGMM(gmm, ax=None, title='GMM', color='b'):
    r"""Plot GMM"""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set(title=title, xlim=[-2, 2], ylim=[-2, 2], aspect='equal')
    for g in gmm.gaussians:
        plot_2d_ellipse(ax, g, color=color, plot_arrows=False)


# TESTS
if __name__ == "__main__":
    from pyrobolearn.distributions.gaussian import plot_2d_ellipse
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # create manually a GMM
    dim, num_components = 2, 3
    gmm = GMM(gaussians=[Gaussian(mean=torch.distributions.Uniform(-1., 1.).sample((dim,)),
                                  covariance=0.1*torch.eye(dim)) for _ in range(num_components)])
    gmm1 = GaussianMixture(n_components=num_components)

    # plot initial GMM
    plotGMM(gmm, title='Initial GMM')
    plt.show()

    # create data
    N, eps = 8, 0.1
    t = torch.linspace(0., 1., 100)
    y = torch.stack([torch.sin(2 * np.pi * t) + eps * torch.rand(len(t)) for _ in range(N)])  # shape: NxT

    # plot data
    plt.plot(t.numpy(), y.t().numpy())
    plt.title('Training data')
    plt.show()

    # combine data (with shape: N'xD, where N' is the N'=N*T)
    X = torch.stack((torch.stack([t] * N).view(-1, 1), y.view(-1, 1)), dim=1)

    # init GMM
    gmm.init(X, method='random')  # method='k-means')
    plotGMM(gmm, title='GMM after K-Means')
    plt.show()

    # fit a GMM using EM
    result = gmm.fit(X, init=None)

    # plot losses
    plt.plot(result['losses'])
    plt.show()

    # plot trained GMM
    fig, ax = plt.subplots(1, 1)
    plotGMM(gmm, ax=ax, title='Trained GMM')
    ax.plot(t, y.t())
    plt.show()

    # fit
    from matplotlib.patches import Ellipse


    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))


    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)


    plot_gmm(gmm1, X)
    plt.show()

    # samples from the GMM and plot

    # GMR: condition on the input variable and plot

    # GMR: condition on the output variable and plot
