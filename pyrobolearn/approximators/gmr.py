#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define Gaussian Process function approximator.

Dependencies:
- `pyrobolearn.models`
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

from pyrobolearn.approximators.approximator import Approximator
from pyrobolearn.models.gmm import GMM


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GMRApproximator(Approximator):
    r"""Gaussian Mixture Regression Approximator

    The Gaussian mixture regression (GMR) approximator depends on the Gaussian mixture model (GMM).


    GMM
    ---

    The Gaussian Mixture Model (GMM) is a semi-parametric, probabilistic and generative model [1,2].
    In robotics, for instance, this is often used to model trajectories by jointly encoding the time and state
    (position and velocity) [3,4,5,6].

    It is mathematically described by:

    .. math:: p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mu_k, \Sigma_k)

    where :math:`K` is the number of components, :math:`\pi_k` are prior probabilities (that is
    :math:`0 \leq \pi_k \leq 1`) that sums to 1 (i.e. :math:`\sum_{k=1}^K \pi_k = 1`),
    :math:`\mathcal{N}(\mu_k, \Sigma_k)` is the multivariate Gaussian (aka Normal) distribution, with mean
    :math:`\mu_k` and covariance :math:`\Sigma_k`. The priors, means, and covariances are grouped to form the
    parameter set :math:`\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K`.


    Learning from data
    ------------------

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


    Gaussian Mixture Regression (i.e. conditioned GMM)
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


    Other miscellaneous information
    -------------------------------

    The conjugate prior of the GMM is the Dirichlet process.


    References:
        - [1] "Pattern Recognition and Machine Learning" (chap 2, 3, 9, and 10), Bishop, 2006
        - [2] "Machine Learning: a Probabilistic Perspective" (chap 3 and 11), Murphy, 2012
        - [3] "Robot Programming by Demonstration: a Probabilistic Approach" (chap 2), Calinon, 2009
        - [4] "A Tutorial on Task-Parameterized Movement Learning and Retrieval", Calinon, 2015
        - [5] "Programming by Demonstration on Riemannian Manifolds" (PhD thesis, chap 1 and 2), Zeerstraten, 2017
        - [6] "Learning Control", Calinon et al., 2018

    """

    def __init__(self, inputs, outputs, num_components=1, priors=None, means=None, covariances=None, gaussians=None,
                 preprocessors=None, postprocessors=None):
        """
        Initialize the Gaussian Mixture regression approximator.

        Args:
            inputs (State, Action, np.array, torch.Tensor): inputs of the inner models (instance of Action/State)
            outputs (State, Action, np.array, torch.Tensor): outputs of the inner models (instance of Action/State)
            num_components (int): the number of components/gaussians (this argument should be provided if
              no priors, means, covariances, or gaussians are provided)
            priors (list/tuple of float, None): prior probabilities (they have to be positives). If not provided,
              it will be a uniform distribution.
            means (list of np.array[float[D]], None): list of means
            covariances (list of np.array[float[D,D]], None): list of covariances
            gaussians (list of Gaussian, None): list of gaussians. If provided, the `means` and `covariances`
              parameters don't have to be provided.
            preprocessors (None, Processor, list of Processor): the inputs are first given to the preprocessors then
              to the model.
            postprocessors (None, Processor, list of Processor): the predicted outputs by the model are given to the
              processors before being returned.
        """
        # create inner model
        num_inputs, num_outputs = self._size(inputs), self._size(outputs)
        dimensionality = num_inputs + num_outputs
        model = GMM(num_components=num_components, priors=priors, means=means, covariances=covariances,
                    gaussians=gaussians, dimensionality=dimensionality)

        # call parent class
        super(GMRApproximator, self).__init__(inputs, outputs, model=model, preprocessors=preprocessors,
                                              postprocessors=postprocessors)
