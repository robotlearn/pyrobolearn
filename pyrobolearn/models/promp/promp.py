#!/usr/bin/env python
"""Define the Probabilistic Movement Primitive class

This file defines the Probabilistic Movement Primitive (ProMP) model, and use the Gaussian distribution defined
in `gaussian.py`.

References
    - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
    - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
"""

from abc import ABCMeta
import numpy as np
# import scipy.interpolate

from pyrobolearn.models.model import Model
from pyrobolearn.models.gaussian import Gaussian
from pyrobolearn.models.promp.canonical_systems import LinearCS
from pyrobolearn.models.promp.basis_functions import BasisMatrix, GaussianBM, VonMisesBM, BlockDiagonalMatrix


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ProMP(object):  # Model
    r"""Probabilistic Movement Primitives

    This class implements the ProMP framework proposed in [1]. This works by putting a prior over the weight
    parameters.

    .. math:: y_t = [q_t, \dot{q}_t]^T = \Phi_t^T w + \epsilon_y

    where :math:`y_t \in \mathbb{R}^{2 \times 1}` is the joint state vector at time step :math:`t`,
    :math:`\Phi_t = [\phi_t, \dot{\phi}_t] \in \mathbb{R}^{M \times 2}` is the matrix containing the basis functions
    defined by the user and where `M` is the number of these basis functions, :math:`w \in \mathbb{R}^{Mx1}` is the
    weight vector on which we put a Gaussian prior distribution given by :math:`w \sim \mathcal{N}(\mu_w, \Sigma_w)`,
    and :math:`epsilon_y \sim \mathcal{N}(0, \Sigma_y)` is the zero-mean Gaussian noise.

    The probability of a specific trajectory :math:`\tau` is given by taking the joint probability distribution, and
    assuming independence between each time step:

    .. math::

        p(\tau | w) = p(y_0, ..., y_T | w) &= \prod_{t=0}^T p(y_t | w) \\
                                           &= \prod_{t=0}^T \mathcal{N}(y_t | \Phi_t^T w, \Sigma_y)

    Note that because we are modeling trajectories with a probability distribution, we need several demonstrations
    in order to capture its variance.

    By marginalizing :math:`p(\tau | w)` such that the specific weights :math:`w` are integrated out, gives us:

    .. math:: p(\tau ; \theta) = \int p(\tau | w) p(w ; \theta) dw

    where :math:`p(w; \theta)` is the Gaussian prior distribution over the weights :math:`w`, and
    :math:`\theta = {\mu_w, \Sigma_w}` are the parameters that describe this distribution. This is referred as the
    marginal likelihood, which is more robust to overfitting as we average over the different models.


    Coupling between Movement Primitives:
    -------------------------------------
    We can generalize the above method and encode the coupling between multiple movement primitives.


    Learning from demonstrations:
    -----------------------------
    In the imitation learning case, when learning from demonstrations, instead of maximizing the likelihood we maximize
    the marginal likelihood (type-II MLE), that is:

    .. math:: \theta^* = \argmax_{\theta} p(\tau ; \theta) = argmax_{\theta} \int p(\tau | w) p(w ; \theta) dw


    Modulation of via-points, final position and velocities by conditioning:
    ------------------------------------------------------------------------
    We can condition the probability on the weights given a desired observation
    :math:`\hat{x}_t = [\hat{y}_t, \hat{\Sigma}_y]`.


    Combination and blending of movement primitives:
    ------------------------------------------------
    * combination
    * blending / sequencing


    References:
        - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
        - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
    """

    __metaclass__ = ABCMeta

    def __init__(self, num_dofs=None, weight_size=None, weights=None, weights_covariance=None,
                 canonical_system=None, noise_covariance=1., Phi=None, promps=None):
        """
        Initialize the probabilistic movement primitive

        Args:
            num_dofs (int, None): number of degrees of freedom.
            weight_size (int, None): size of the weight mean vector. This has to be specified if one of the following
                arguments are not given: `Phi`, `weights`, `weights_covariance`.
            weights (None, np.array[DM], Gaussian): mean of the weigths, or Gaussian weight distribution
            weights_covariance (None): covariance on the weights
            canonical_system (None, CS): canonical system
            noise_covariance (float, np.array[2Dx2D]): covariance on the noise
            Phi (None, BasisMatrix[DM,2D]): callable basis matrix
            promps (list[ProMP], None): list of ProMPs (useful when combining different ProMPs)
        """
        super(ProMP, self).__init__()

        # check Phi # TODO: check shape
        if Phi is not None and not isinstance(Phi, BasisMatrix):
            raise TypeError("Expecting the given basis matrix Phi to be an instance of `BasisMatrix`, instead got: "
                            "{}".format(Phi))

        # check if multiple promps are given
        if promps:
            # Notes: the ProMPs can be of different type, and/or have different nb of basis fcts

            # check that we are given a list of promps
            if not isinstance(promps, (list, tuple)):
                raise TypeError("Expecting a list of ProMPs for `promps`")
            for i, promp in enumerate(promps):
                if not isinstance(promp, ProMP):
                    raise TypeError("The item {} is not an instance of ProMP".format(i))

            # check other attributes
            for i in range(0, len(promps) - 1):
                promp, next_promp = promps[i], promps[i+1]

                # check if the basis matrix has been defined
                if promp.Phi is None:
                    raise ValueError("The ProMP {} doesn't have a basis matrix defined".format(i))

                # check if each ProMP has the same dimensionality (i.e. the same number of DoF)
                if promp.dim != next_promp.dim:
                    raise ValueError("The ProMPs {} and {} have different dimensions".format(i, i+1))

                # check if each ProMP has the same number of time steps
                if promp.cs.num_timesteps != next_promp.cs.num_timesteps:
                    raise ValueError("The ProMPs {} and {} have different number of time steps".format(i, i+1))

            num_dofs = promps[0].dim

        # else, define one promp
        else:
            if Phi is not None:
                # basis matrix (shape should be: DMx2D where D=nb of DoFs, and M=nb of basis fcts)
                num_dofs = Phi.shape[1] / 2
            else:
                if num_dofs is None:
                    raise ValueError("The number of degrees of freedom should be specified")

            # quick checks
            if num_dofs < 1:
                raise ValueError("Expecting at least one degree freedom")

            # create Gaussian weight distribution (with shape(mean) = DM, shape(cov) = DMxDM)
            if isinstance(weights, Gaussian):
                # check that it matches with the dimension of Phi if specified
                if Phi is not None:
                    if Phi.shape[0] != weights.mean.size:
                        raise ValueError("Mismatch between the dimensions of Phi and the weights; got Phi.shape[0]={} "
                                         "and weights.size={}".format(Phi.shape[0], weights.mean.size))
            else:
                # infer the size of the weights
                if Phi is not None:
                    size = Phi.shape[0]
                    if weights:
                        if weights.size != size:
                            raise ValueError(
                                "Mismatch between the dimensions of Phi and the weights; got Phi.shape[0]={} "
                                "and weights.size={}".format(size, weights.size))
                elif weights:
                    size = weights.size
                elif weights_covariance:
                    size = weights_covariance.shape[0]
                else:
                    size = weight_size

                # check that the size is valid
                if not isinstance(size, int):
                    raise TypeError("Expecting the weight_size to be given and to be an integer")
                if size <= 0:
                    raise ValueError("Expecting the size to be positive")

                # create mean vector and covariance matrix if no weigths specified
                if weights is None:
                    weights = np.random.rand(size)
                if weights_covariance is None:
                    weights_covariance = np.identity(size)

                if weights_covariance.shape != (size, size):
                    raise ValueError("Expecting a square matrix for the covariance matrix of shape SxS, where S is "
                                     "the size of the weight vector")

                # create weight distribution
                weights = Gaussian(weights, weights_covariance)

            # create Gaussian noise distribution (with shape(mean) = 2D, shape(cov) = 2Dx2D)
            if isinstance(noise_covariance, (float, int)):
                noise_covariance = noise_covariance * np.identity(2 * num_dofs)
            self._noise = Gaussian(np.zeros(2 * num_dofs), noise_covariance)

        # set the variables
        self.D = num_dofs
        self._weights = weights
        self.promps = promps
        self.Phi = Phi      # shape: DMx2D

        # priority exponent (only play a role if combining different ProMPs)
        self.priority = 1.

        # create canonical system
        self.cs = canonical_system if canonical_system is not None else LinearCS()

    ##############
    # Properties #
    ##############

    @property
    def canonical_system(self):
        """Return the canonical system"""
        return self.cs

    @property
    def num_mps(self):
        """Return the number of ProMPs."""
        if self.promps:
            return len(self.promps)
        return 1

    # alias
    num_promps = num_mps

    @property
    def num_dofs(self):
        """Return the number of degrees of freedom"""
        return self.D

    @property
    def dim(self):
        """Return the dimensionality which is 2 * the number of degrees of freedom (2 because we have position and
        velocity info)"""
        return 2 * self.D

    # TODO: generalize it with different number of basis functions
    @property
    def num_basis_per_dof(self):
        """Return the number of basis function per degree of freedom"""
        return self.Phi.num_basis

    @property
    def total_num_basis(self):
        """Return the total number of basis functions"""
        return self.Phi.shape[0] / self.num_dofs

    @property
    def basis_matrix(self):
        """Return the basis matrix"""
        return self.Phi

    @property
    def weights(self):
        """Return the weight distribution"""
        return self._weights

    @property
    def noise(self):
        """Return the noise distribution"""
        return self._noise

    @property
    def input_dims(self):
        """Return the input dimension of the model, which is one as it only accepts the phase variable"""
        return 1

    @property
    def output_dims(self):
        """Return the output dimension of the model"""
        return 2 * self.D

    @property
    def priority(self):
        """Return the priority exponent"""
        return self._alpha

    @property
    def priority_fct(self):
        """Return the priority exponent function"""
        return self._alpha_fct

    @priority.setter
    def priority(self, priority):
        """Set the priority exponent which is a function"""
        if priority is None:
            self._alpha = 1
        elif isinstance(priority, (float, int)):
            if priority < 0 or priority > 1:
                raise ValueError("Priority exponents can only be between 0 and 1")
            self._alpha = priority
            self._alpha_fct = lambda s: priority
        # elif isinstance(priority, np.ndarray):
        #     if len(priority) != self.cs.num_timesteps:
        #         raise ValueError("Mismatch between the number of priorities and the number of phases. Here, we "
        #                          "assume the user wants to give a priority for each phase.")
        #     if np.all(priority < 0) or np.all(priority > 1):
        #         raise ValueError("Some priority exponents are not between 0 and 1")
        #     self._alpha = priority
        elif callable(priority):
            # check if the callable priority accepts the phase value(s)
            try:
                # check if the callable priority accepts the phase values one by one
                self.cs.reset()
                for _ in range(self.cs.num_timesteps):  # for one period
                    s = self.cs.step()
                    p = priority(s)
                    if not isinstance(p, (float, int)):
                        raise TypeError("The callable function/class doesn't return a float/int number given "
                                        "the phase value.")
                    if p < 0 or p > 1:
                        raise ValueError("The callable function returned a priority which is not between 0 and 1; "
                                         "given the phase value s={}, it returned the priority p={}".format(s, p))

                # check if the callable priority accepts the phase values all at once
                self.cs.reset()
                s = self.cs.rollout()
                p = priority(s)
                if not isinstance(p, np.ndarray):
                    raise TypeError("The callable function/class doesn't return an np.array given the phase values")
                if np.all(p < 0) or np.all(p > 1):
                    raise ValueError("Some priority exponents returned by the callable function/class are not between "
                                     "0 and 1")

                # reset the canonical system
                self.cs.reset()
            except:
                raise ValueError("The callable function/class doesn't accept float numbers or ")
            self._alpha = priority
            self._alpha_fct = priority
        else:
            raise TypeError("Priority exponent can only be None, a float, or an array of floats")

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other):
        """Return a copy of a ProMP"""
        if not isinstance(other, ProMP):
            raise TypeError("Trying to copy an object which is not a ProMP")
        pass

    @staticmethod
    def is_parametric():
        """The ProMP is a parametric model"""
        return True

    @staticmethod
    def is_linear():
        """The ProMP has linear parameters"""
        return True

    @staticmethod
    def is_recurrent():
        """The ProMP is not a recurrent model where outputs depends on previous inputs. Its sequential nature
        is due to the fact that the time is given as an input to the ProMP model"""
        return False

    @staticmethod
    def is_probabilistic():
        """The ProMP is a probabilistic model which parametrizes a normal distribution (by specifying its mean
        and covariance)"""
        return True

    @staticmethod
    def is_discriminative():
        r"""The ProMP is a discriminative model which predicts :math:`p(y|x)` where :math:`x` is the (time) input,
        and :math:`y` is the output"""
        return True

    @staticmethod
    def is_generative():
        """The ProMP is not a generative model, and thus we can not sample from it"""
        return False

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Returns an iterator over the model parameters."""
        # yield self.weights
        yield self.weights.mean
        yield self.weights.cov

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        # yield "Gaussian", self.weights
        yield "mean", self.weights.mean
        yield "covariance", self.weights.cov

    def reset(self):
        """Reset the canonical systems"""
        self.cs.reset()

    def basis(self, s):
        r"""
        Return the basis matrix evaluated at the given phase(s).

        Args:
            s (np.array[T], float): phase value(s)

        Returns:
            np.array[DM,2D], np.array[DM,T,2D]: basis matrix evaluated at the given phase(s).
        """
        return self.Phi(s)

    def weighted_basis(self, s):
        r"""
        Return the weighted basis evaluated at the given phase(s), that is, `Phi(s) * w`, where `w` is the weight
        mean vector, and the product is a broadcast product (and not a dot/scalar product).

        Args:
            s (np.array[T], float): phase value(s)

        Returns:
            np.array[DM,T,2D]: weighted basis matrix (this has the same shape as the basis matrix Phi(s))
        """
        return (self.Phi(s).T * self.weights.mean).T

    def predict_conditional(self, s, weights=None):
        r"""
        Predict conditional output distribution given the weights :math:`p(y_t|w)`.

        This is given by:

        .. math:: p(y_t|w) = \mathcal{N}(\Phi(s_t)^\top \mu_w , \Sigma_y)

        Args:
            s (float, np.array[T]): phase value(s)
            weights (None, np.array[DM]): weights (array). If None, it will take the mean/mode of the weight
                distribution.

        Returns:
            Gaussian or list[Gaussian]: conditional output distribution for each phase value
        """
        # if multiple promps that need to be combined
        if self.promps:
            gaussians = [promp.predict_conditional(s) for promp in self.promps]
            priorities = [promp.priority_fct(s) for promp in self.promps]

            # if one phase value
            if isinstance(gaussians[0], Gaussian):
                covs, means = [], []
                for g, p in zip(gaussians, priorities):
                    # compute covariance and mean
                    prec = g.prec
                    cov = p * prec
                    mean = p * prec.dot(g.mean)
                    covs.append(cov)
                    means.append(mean)

                cov = np.linalg.inv(np.sum(covs, axis=0))
                mean = cov.dot(np.sum(means, axis=0))

                return Gaussian(mean=mean, covariance=cov)

            # if multiple phases
            else:
                gaussian_results = []
                for gaussian, priority in zip(gaussians, priorities):
                    covs, means = [], []
                    for g, p in zip(gaussian, priority):  # zip(gaussians, priorities)
                        # compute covariance and mean
                        prec = g.prec
                        cov = p * prec
                        mean = p * prec.dot(g.mean)
                        covs.append(cov)
                        means.append(mean)

                    cov = np.linalg.inv(np.sum(covs, axis=0))
                    mean = cov.dot(np.sum(means, axis=0))
                    g = Gaussian(mean=mean, covariance=cov)
                    gaussian_results.append(g)

                return gaussian_results

        # if one promp
        if weights is None:
            weights = self.weights.mean
        if isinstance(s, (float, int)):
            return Gaussian(mean=self.Phi(s).T.dot(weights), covariance=self.noise.cov)
        return [Gaussian(mean=self.Phi(phase).T.dot(weights), covariance=self.noise.cov) for phase in s]

    def predict_marginal(self, s):
        r"""
        Predict marginal output distribution :math:`p(y_t; \theta)`.

        .. math::

            p(y_t; \theta) &= \int p(y_t | w) p(w; \theta) dw \\
                           &= \int \mathcal{N}(y_t | \Phi(s_t)^\top w, \Sigma_y) \mathcal{N}(w | \mu_w, \Sigma_w) dw
                           &= \mathcal{N}(\Phi(s_t)^\top \mu_w, \Phi(s_t)^\top \Sigma_w \Phi(s_t) + \Sigma_y)

        where :math:`\theta = \{\mu_w, \Sigma_w\}` are the parameters.

        Args:
            s (float, np.array[T]): phase value(s)

        Returns:
            Gaussian or list[Gaussian]: marginal output distribution for each phase value
        """
        # if multiple promps that need to be combined
        if self.promps:
            gaussians = [promp.predict_marginal(s) for promp in self.promps]
            priorities = [promp.priority_fct(s) for promp in self.promps]

            # if one phase value
            if isinstance(gaussians[0], Gaussian):
                covs, means = [], []
                for g, p in zip(gaussians, priorities):
                    # compute covariance and mean
                    prec = g.prec
                    cov = p * prec
                    mean = p * prec.dot(g.mean)
                    covs.append(cov)
                    means.append(mean)

                cov = np.linalg.inv(np.sum(covs, axis=0))
                mean = cov.dot(np.sum(means, axis=0))

                return Gaussian(mean=mean, covariance=cov)

            # if multiple phases
            else:
                gaussian_results = []
                for gaussian, priority in zip(gaussians, priorities):
                    covs, means = [], []
                    for g, p in zip(gaussians, priorities):
                        # compute covariance and mean
                        prec = g.prec
                        cov = p * prec
                        mean = p * prec.dot(g.mean)
                        covs.append(cov)
                        means.append(mean)

                    cov = np.linalg.inv(np.sum(covs, axis=0))
                    mean = cov.dot(np.sum(means, axis=0))
                    g = Gaussian(mean=mean, covariance=cov)
                    gaussian_results.append(g)

                return gaussian_results

        # if one promp
        if isinstance(s, (float, int)):
            return self.Phi(s).T * self.weights + self.noise
        return [(self.Phi(phase).T * self.weights + self.noise) for phase in s]

    def predict(self, s, sample=False, method='conditional'):
        r"""
        Predict output mean :math:`\mu_y` given input data :math:`s`.

        .. math:: \mu_y = \Phi(s)^T \mu_w + \epsilon_y

        where :math:`\Phi(s)` is the basis matrix, :math:`w` is the weight vector, and :math:`\epsilon_y` is the
        Gaussian noise such that :math:`\epsilon_y \sim \mathcal{N}(0, \Sigma_y)`

        Args:
            s (float, np.array[T]): phase value(s)
            method (str): choice between 'conditional' and 'marginal' prediction
            sample (bool): if False, it will return the mode/mean of the Gaussian, otherwise it will sample from it
                for each phase value

        Returns:
            np.array[2D], np.array[T,2D]: output mean(s)
        """
        # Prediction
        if method == 'conditional':
            gaussians = self.predict_conditional(s)
        elif method == 'marginal':
            gaussians = self.predict_marginal(s)
        else:
            raise NotImplementedError("The given 'method' argument has not been implemented. Please choose "
                                      "between 'conditional' and 'marginal'.")

        # if one phase value, return predicted output vector (shape: 2D)
        if isinstance(gaussians, Gaussian):
            if sample:
                return gaussians.sample()
            return gaussians.mode

        # if multiple phase values, return output vectors (shape: [T,2D])
        else:
            if sample:
                return np.array([g.sample() for g in gaussians])
            return np.array([g.mode for g in gaussians])

    def predict_proba(self, s, method='marginal', return_gaussian=True):
        r"""
        Predict the probability of output :math:`\mathcal{N}(\mu_y, \Sigma_y)` given input data :math:`s`.

        .. math::

            p(y; \theta) &= \int p(y | w) p(w; \theta) dw \\
                         &= \int \mathcal{N}(y | \Phi(s)^\top w, \Sigma_y) \mathcal{N}(w | \mu_w, \Sigma_w) dw
                         &= \mathcal{N}(\Phi(s)^\top \mu_w, \Phi(s)^\top \Sigma_w \Phi(s) + \Sigma_y)

        where :math:`\theta = \{\mu_w, \Sigma_w\}` are the parameters.

        Args:
            s (float, np.array[T]): phase value(s)
            method (str): choice between 'conditional' and 'marginal' prediction
            return_gaussian (bool): If True, it will return a Gaussian for each input phase value. If False,
                it will return the mean and covariance for each phase value.

        Returns:
            if return_gaussian:
                Gaussian, or list[Gaussian]: gaussian(s) for each input phase value
            else:
                np.array[2D], np.array[T,2D]: output mean(s)
                np.array[2D,2D], np.array[T,2D,2D]: output covariance(s)
        """
        # Predict
        if method == 'conditional':
            y = self.predict_conditional(s)
        elif method == 'marginal':
            y = self.predict_marginal(s)
        else:
            raise NotImplementedError("The given 'method' argument has not been implemented. Please choose "
                                      "between 'conditional' and 'marginal'.")

        # return Gaussian in the desired form
        if return_gaussian:
            return y

        # if one phase value
        if isinstance(y, Gaussian):
            return y.mean, y.cov

        # if multiple phase values
        return np.array([gaussian.mean for gaussian in y]), np.array([gaussian.cov for gaussian in y])

    def step(self, tau=1., sample=False, method='marginal'):
        r"""
        Perform one step forward with the canonical system, and return the deterministic predicted output, given by:

        .. math:: y = \Phi(s)^T \mu_w + \epsilon_y

        where :math:`\Phi(s)` is the basis matrix, :math:`w` is the weight vector, and :math:`\epsilon_y` is the
        noise.

        Args:
            tau (float): speed
            sample (bool): if False, it will return the mode/mean of the Gaussian, otherwise it will sample from it
                for each phase value.
            method (str): choice between 'conditional' and 'marginal' prediction

        Returns:
            np.array[2D]: predicted output
        """
        s = self.cs.step(tau=tau)
        return self.predict(s, sample=sample, method=method)

    def step_proba(self, tau=1., method='marginal', return_gaussian=True):
        r"""
        Perform one step forward with the canonical system, and return the predicted output distribution
        :math:`p(y_t | w)` and :math:`p(y_t ; \theta)` based on the specified `method`.

        Args:
            tau (float): speed
            method (str): choice between 'conditional' and 'marginal' prediction
            return_gaussian (bool): If True, it will return a Gaussian for the current phase value. If False,
                it will return the mean and covariance of the current phase value.

        Returns:
            if return_gaussian:
                Gaussian: gaussian for the phase value
            else:
                np.array[2D]: output mean
                np.array[2D,2D]: output covariance
        """
        s = self.cs.step(tau=tau)
        return self.predict_proba(s, method=method, return_gaussian=return_gaussian)

    def rollout(self, tau=1., sample=False, method='marginal'):
        r"""
        Perform a complete rollout; predict a whole trajectory from the initial phase to final one.

        Args:
            tau (float): speed
            sample (bool): if False, it will return the mode/mean of the Gaussian, otherwise it will sample from it
                for each phase value.
            method (str): choice between 'conditional' and 'marginal' prediction

        Returns:
            np.array[T,2D]: predicted outputs
        """
        # reset system
        self.reset()

        # rollout with the canonical system
        s = self.cs.rollout(tau=tau)

        # return predictions
        return self.predict(s, sample=sample, method=method)

    def rollout_proba(self, tau=1., method='marginal', return_gaussian=True):
        r"""
        Perform a complete probabilistic rollout; predict a whole probabilistic trajectory from the initial phase to
        final one.

        Args:
            tau (float): speed
            method (str): choice between 'conditional' and 'marginal' prediction
            return_gaussian (bool): If True, it will return a Gaussian for each phase value. If False,
                it will return the mean and covariance for each phase value.

        Returns:
            if return_gaussian:
                list[Gaussian]: gaussians for each phase value
            else:
                np.array[T,2D]: output means
                np.array[T,2D,2D]: output covariances
        """
        # reset system
        self.reset()

        # rollout with the canonical system
        s = self.cs.rollout(tau=tau)

        # return predictions
        return self.predict_proba(s, method=method, return_gaussian=return_gaussian)

    def sample_weights(self, size=None, seed=None):
        """
        Sample weight vector from the distribution.

        Args:
            size (int, None): number of samples
            seed (int, None): seed for the random number generator

        Returns:
            np.array[DM], np.array[N,DM]: samples
        """
        return self.weights.sample(size=size, seed=seed)

    def sample_trajectory(self, size=None):
        r"""
        Sample one complete trajectory.

        .. math:: \tau \sim p(\tau; \theta) = \prod_{t=1}^T p(y_t; \theta)

        Warnings: we assume complete independence between the :math:`y_t` instead of a conditional independence
        of the :math:`y_t` given the weights :math:`w`.

        Args:
            size (int, None): number of samples

        Returns:
            np.array[T,2D], np.array[N,T,2D]: trajectory/trajectories
        """
        if size is None or size < 2:
            return self.rollout(tau=1., sample=True, method='marginal')
        return np.array([self.rollout(tau=1., sample=True, method='marginal') for _ in range(size)])

    def sample_from_prediction(self, s, size=None, seed=None, method='conditional'):
        r"""
        Sample from predicted output distribution :math:`y_t \sim p(y_t | w)` or :math:`y_t \sim p(y_t; \theta)`,
        based on the specified `method`.

        Args:
            s (float): phase value
            size (int, None): number of samples
            seed (int, None): seed for the random number generator
            method (str): choice between 'conditional' and 'marginal' prediction

        Returns:
            np.array[2D], np.array[N,2D]: sample(s)
        """
        if not isinstance(s, (float, int)):
            raise TypeError("Expecting only one phase value (float)")
        gaussian = self.predict_proba(s, method=method, return_gaussian=True)
        return gaussian.sample(size=size, seed=seed)

    def sample_from_conditional(self, s, size=None, seed=None):
        r"""
        Sample output from the likelihood (conditional probability).

        .. math:: y_t \sim p(y_t | w)

        where :math:`w` are the weights. Because :math:`y_t = \Phi(s_t)^\top w + \epsilon_y` with :math:`\epsilon_y`
        being a zero-mean Gaussian noise (i.e. :math:`\epsilon_y \sim \mathcal{N}(0, \Sigma_y)`), the outputs are
        sampled from the following distribution:

        .. math:: y_t \sim \mathcal{N}(\Phi(s)^\top w, \Sigma_y)

        Args:
            s (float): phase value
            size (int, None): number of samples
            seed (int, None): seed for the random number generator

        Returns:
            np.array[2D], np.array[N,2D]: sample(s)
        """
        if not isinstance(s, (float, int)):
            raise TypeError("Expecting only one phase value (float)")
        gaussian = self.predict_conditional(s)
        return gaussian.sample(size=size, seed=seed)

    def sample_from_marginal_likelihood(self, s, size=None, seed=None):
        r"""
        Sample output from the marginal likelihood.

        .. math:: y_t \sim p(y_t; \theta)

        where :math:`\theta = {\mu_w, \Sigma_w}` are the parameters of the Gaussian distribution put on the weights,
        and :math:`p(y_t; \theta) = \int p(y_t | w) p(w; \theta) dw`. The outputs are thus sampled from:

        .. math:: y_t \sim \mathcal{N}(\Phi(s)^\top \mu_w, \Phi(s)^\top \Sigma_w \Phi(s) + \Sigma_y)

        Args:
            s (float): phase value
            size (int, None): number of samples
            seed (int, None): seed for the random number generator

        Returns:
            np.array[2D], np.array[N,2D]: sample(s)
        """
        if not isinstance(s, (float, int)):
            raise TypeError("Expecting only one phase value (float)")
        gaussian = self.predict_marginal(s)
        return gaussian.sample(size=size, seed=seed)

    def likelihood(self, y, s=None):
        r"""
        Compute the likelihood of the output data given the input data.

        .. math:: p(y_{1:T} | w) = \prod_{t=1}^T \mathcal{N}(y_t | \Phi(s_t)^\top \mu_w, \Sigma_y)

        Args:
            y (np.array[2D], np.array[T,2D]): output vector(s) to evaluate the likelihood. It can be an output
                vector at one particular time step, or an output vector for each time step
            s (float, np.array[T], None): the corresponding time step(s). If None, it will generate the same number
                of phase values than the number of output vectors, and uniformly place them between [0,1].

        Returns:
            float: likelihood
        """
        if len(y.shape) == 1:
            if not isinstance(s, (float, int)):
                raise TypeError("One sample is given but not its associated phase value (float number)")
            gaussian = self.predict_conditional(s)
            return gaussian.pdf(y)
        if len(y.shape) != 2:
            raise ValueError("Expecting a 2D matrix containing the concatenated output vectors")
        T = y.shape[0]
        if s is None:
            s = np.linspace(0.,1.,T)
        if isinstance(s, (float, int)):
            s = s * np.ones(T)
        if len(s) != T:
            raise ValueError("Expecting the number of phase values to be the same as the number of output vectors")
        gaussians = self.predict_conditional(s)
        return np.prod([gaussian.pdf(yt) for yt, gaussian in zip(y, gaussians)])

    # alias
    pdf = likelihood

    def log_likelihood(self, y, s=None):
        r"""
        Compute the log-likelihood.

        .. math:: \log p(y_{1:T} | w) = \sum_{t=1}^T \log \mathcal{N}(y_t | \Phi(s_t)^\top \mu_w, \Sigma_y)

        Args:
            y (np.array[2D], np.array[T,2D]): output vector(s) to evaluate the likelihood. It can be an output
                vector at one particular time step, or an output vector for each time step
            s (float, np.array[T], None): the corresponding time step(s). If None, it will generate the same number
                of phase values than the number of output vectors, and uniformly place them between [0,1].

        Returns:
            float: log-likelihood
        """
        return np.log(self.likelihood(y, s))

    # alias
    log_pdf = log_likelihood

    def marginal_likelihood(self, y, s=None):
        r"""
        Compute the marginal likelihood (which is the loss being optimized in ProMPs) by assuming that we have
        independence between the output vectors :math:`y_t`.

        If we assume independence between each sample :math:`y_t` then:

        .. math::

            p(y_{1:T}; \theta) = \prod_{t=1}^T p(y_t; \theta)

        where

        .. math::

            p(y_t; \theta) &= \int p(y_t | w) p(w; \theta) dw \\
                           &= \int \mathcal{N}(y_t | \Phi(s_t)^\top w, \Sigma_y) \mathcal{N}(w | \mu_w, \Sigma_w) dw
                           &= \mathcal{N}(y_t | \Phi(s_t)^\top \mu_w, \Phi(s_t)^\top \Sigma_w \Phi(s_t) + \Sigma_y)

        where :math:`\theta = \{\mu_w, \Sigma_w\}` are the parameters.

        Note that if instead of independence, we assume conditional independence given the weights :math:`w`,
        then we have:

        .. math::

            p(y_{1:T}; \theta) &= \int p(y_{1:T} | w) p(w; \theta) dw \\
                               &= \int \prod_{t=1}^T p(y_t | w) p(w; \theta) dw

        Note that in this expression the product is inside the integral instead of outside.

        Args:
            y (np.array[2D], np.array[T,2D]): output vector(s) to evaluate the likelihood. It can be an output
                vector at one particular time step, or an output vector for each time step
            s (float, np.array[T], None): the corresponding time step(s). If None, it will generate the same number
                of phase values than the number of output vectors, and uniformly place them between [0,1].

        Returns:
            float: marginal likelihood
        """
        if len(y.shape) == 1:
            if not isinstance(s, (float, int)):
                raise TypeError("One sample is given but not its associated phase value (float number)")
            gaussian = self.predict_marginal(s)
            return gaussian.pdf(y)
        if len(y.shape) != 2:
            raise ValueError("Expecting a 2D matrix containing the concatenated output vectors")
        T = y.shape[0]
        if s is None:
            s = np.linspace(0.,1.,T)
        if isinstance(s, (float, int)):
            s = s * np.ones(T)
        if len(s) != T:
            raise ValueError("Expecting the number of phase values to be the same as the number of output vectors")
        gaussians = self.predict_marginal(s)
        return np.prod([gaussian.pdf(yt) for yt, gaussian in zip(y, gaussians)])

    def log_marginal_likelihood(self, y, s):
        r"""
        Compute the marginal log-likelihood (which is the loss being optimized in ProMPs).

        If we assume independence between each sample :math:`y_t` then:

        .. math::

            \log p(y_{1:T}; \theta) = \sum_{t=1}^T \log p(y_t; \theta)

        See also the `marginal_likelihood` method for more information.

        Args:
            y (np.array[2D], np.array[T,2D]): output vector(s) to evaluate the likelihood. It can be an output
                vector at one particular time step, or an output vector for each time step
            s (float, np.array[T], None): the corresponding time step(s). If None, it will generate the same number
                of phase values than the number of output vectors, and uniformly place them between [0,1].

        Returns:
            float: log marginal likelihood
        """
        return np.log(self.marginal_likelihood(y, s))

    def joint_distribution(self, y, Phi_or_s, y_cov=None):
        r"""
        Compute and return the joint distribution between the weights and the output vector :math:`p(w, y)`. This will
        be useful when computing the posterior conditional probability of the weights given the output vector
        :math:`p(w|y)`.

        .. math:: p(w, y) = \mathcal{N}(\mu, \Sigma)

        where :math:`\mu = [\mu_w^\top, (\Phi^\top \mu_w)^\top]^\top` and :math:`\Sigma = \left[ \begin{array}{cc}
        \Sigma_w & \Sigma_w \Phi \\ \Phi^\top \Sigma_w & \Phi^\top \Sigma_w \Phi + \Sigma_y \end{array} \right]`, with
        :math:`\Sigma_y` is the specified covariance for the output vector. If it is not provided, it will be set to 0.

        Args:
            y (np.array[2D]): output vector
            Phi_or_s (np.array[DM,2D], float): basis matrix evaluated at a particular phase, or phase value.
                If the phase is given, the basis matrix will computed internally.
            y_cov (np.array[2D,2D]): desired covariance.

        Returns:
            Gaussian: joint Gaussian distribution between the output and weights
        """
        # Quick checks
        y_cov_shape = (y.shape[-1], y.shape[-1])
        if y_cov is None:
            y_cov = np.zeros(y_cov_shape)
        if y_cov.shape != y_cov_shape:
            raise ValueError("Expecting a 2D array of shape {} for the output covariance".format(y_cov_shape))

        # check if phase value or basis matrix given
        if isinstance(Phi_or_s, (float, int)): # phase value
            Phi = self.Phi(Phi_or_s)
        else:
            Phi = Phi_or_s

        # compute joint distribution between weights and output vector
        w_mean, w_cov = self.weights.mean, self.weights.cov  # shape: DM and DMxDM
        joint_mean = np.concatenate((w_mean, Phi.T.dot(w_mean)))
        joint_cov = np.vstack((np.hstack((w_cov, w_cov.dot(Phi))),
                               np.hstack((Phi.T.dot(w_cov), Phi.T.dot(w_cov).dot(Phi) + y_cov))))

        # return joint distribution
        return Gaussian(mean=joint_mean, covariance=joint_cov)

    def posterior_weights(self, y, Phi_or_s, y_cov=None):
        r"""
        Compute the posterior distribution on the weights given the output vector :math:`p(w|y)`.

        .. math:: p(w|y) = \mathcal{\mu_w', \Sigma_w'}

        where :math:`\mathcal{\mu_w', \Sigma_w'}` is the resulting conditional Gaussian distribution, with:

        .. math::

            \mu_w' &= \mu_w + \Sigma_w \Phi \Sigma_{yy}^{-1} (y - \Phi^\top \mu_w) \\
            \Sigma_w' &= \Sigma_w - \Sigma_w \Phi \Sigma_{yy}^{-1} \Phi^\top \Sigma_w

        with :math:`\Sigma_{yy} = \Phi^\top \Sigma_w \Phi + \Sigma_y` where :math:`\Sigma_y` is the specified
        covariance for the output vector. If it is not provided, it will be set to 0.

        Args:
            y (np.array[2D]): output vector
            Phi_or_s (np.array[DM,2D], float): basis matrix evaluated at a particular phase, or phase value.
                If the phase is given, the basis matrix will computed internally.
            y_cov (np.array[2D,2D]): desired covariance.

        Returns:
            Gaussian: posterior distribution on the weights given the output vector (along with its possible
                desired covariance)
        """
        # check if phase value or basis matrix given
        if isinstance(Phi_or_s, (float, int)):  # phase value
            Phi = self.Phi(Phi_or_s)
        else:
            Phi = Phi_or_s

        # compute joint distribution between weight and given output vector
        joint = self.joint_distribution(y, Phi, y_cov)

        # compute the posterior weight by conditioning the joint Gaussian
        weight_idx_in_joint = range(self.weights.mean.size)
        weight = joint.condition(y, weight_idx_in_joint)

        return weight

    def compute_loss(self, Y, kind='marginal_log_likelihood'):
        r"""
        Compute the loss on the whole given data.

        Args:
            Y (np.array[N,T,2D], list[np.array[T,2D]]): state trajectories
            kind (str): specifies the kind/type of loss we wish to compute. Select between 'likelihood',
                'log_likelihood', 'marginal_likelihood', 'marginal_log_likelihood'

        Returns:
            float: loss
        """
        # compute phases
        phases = [np.linspace(0., 1., len(y)) for y in Y]

        if kind == 'marginal_log_likelihood':
            return np.sum([self.log_marginal_likelihood(y, s) for y, s in zip(Y, phases)])
        elif kind == 'log_likelihood':
            return np.sum([self.log_likelihood(y, s) for y, s in zip(Y, phases)])
        elif kind == 'likelihood':
            return np.sum([self.likelihood(y, s) for y, s in zip(Y, phases)])
        elif kind == 'marginal_likelihood':
            return np.sum([self.marginal_likelihood(y, s) for y, s in zip(Y, phases)])
        else:
            raise ValueError("The specified kind of loss has not been implemented")

    def expectation_maximization(self, Y, num_iters=1000, threshold=1e-4, verbose=False):
        r"""
        Learn the parameters :math:`\theta = \{\mu_w, \Sigma_w\}` of the Gaussian distribution put on the weights.

        Args:
            Y (np.array[N,T,2D], list[np.array[T,2D]]): state trajectories
            num_iters (int): number of iterations for the EM algo
            threshold (float): convergence threshold for the EM algo
            verbose (bool): if we should print details during the optimization process

        Returns:
            dict: dictionary containing info collected during the optimization process, such as the history of losses,
                the number of iterations it took to converge, if it succeeded, etc.
        """
        # TODO: quick checks

        # compute dictionary results
        results = {'losses': [], 'success': False, 'num_iters': 0}

        # compute initial loss
        loss = self.compute_loss(Y, kind='marginal_log_likelihood')
        prev_loss = loss
        results['losses'].append(loss)

        for it in range(num_iters):
            # E-step: posterior distribution on weights
            means, covs = [], []
            for y in Y:
                # compute phase
                T = len(y)
                phases = np.linspace(0., 1., T)

                # compute weight vector
                y = y.reshape(-1)  # shape: 2DT
                Phi = np.hstack([self.Phi(s) for s in phases])  # DMx2DT
                weight = self.posterior_weights(y, Phi)  # Gaussian with mean of shape DM

                # append mean and cov
                means.append(weight.mean)
                covs.append(weight.cov)

            means = np.array(means)     # shape: NxDM
            covs = np.array(covs)       # shape: NxDMxDM

            # M-step: result from optimizing the complete-data log-likelihood
            mean = Gaussian.compute_mean(means)     # shape: DM
            cov = Gaussian.compute_covariance(means - mean, bessels_correction=False)    # shape: DMxDM
            cov += Gaussian.compute_mean(covs, axis=0)      # shape: DMxDM
            # set new weights for ProMP
            self._weights = Gaussian(mean=mean, covariance=cov)

            # 4. check convergence
            self.compute_loss(Y, kind='marginal_log_likelihood')
            results['losses'].append(loss)
            if np.abs(loss - prev_loss) <= threshold:
                if verbose:
                    print("Convergence achieved at iteration {} with associated loss: {}".format(it + 1, loss))
                results['num_iters'] = it + 1
                results['success'] = True

            # update previous loss
            prev_loss = loss

        return results

    # TODO: implement following method
    def maximum_marginal_likelihood(self, Y, prior_reg=1., num_iters=1000, threshold=1e-4, verbose=False):
        r"""
        Compute the closed-form solutions for the mean and covariance that maximizes the marginal likelihood.

        Assuming independence in time between the predicted output, we have
        :math:`p(y_{t=1:T} ; \theta) = \prod_{t=1}^T p(y_t ; \theta)`. Also, by assuming iid sampled trajectories,
        the log marginal loss to be optimized is given by:

        .. math:: \mathcal{L}(\theta) = \log p(Y; \theta) = \sum_{n=1}^N \sum_{t=1}^T \log p(y_t^{(n)}; \theta)

        Args:
            Y (np.array[N,T,2D], list[np.array[T,2D]]): state trajectories
            prior_reg (float): prior regularization
            num_iters (int): number of iterations for the EM algo
            threshold (float): convergence threshold for the EM algo
            verbose (bool): if we should print details during the optimization process

        Returns:
            dict: dictionary containing info collected during the optimization process, such as the history of losses,
                the number of iterations it took to converge, if it succeeded, etc.

        References:
            [1] "The Matrix Cookbook", Petersen and Pedersen, 2012
            [2] "Second Order Adjoint Matrix Equation", Crone, 1981
        """
        results = {'losses': [], 'success': False, 'num_iters': 1}
        raise NotImplementedError
        # return results

    def linear_ridge_regression(self, Y, prior_reg=1.):
        r"""
        Learn the weights for the ProMP using linear ridge regression.

        Args:
            Y (np.array[N,T,2D], list[np.array[T,2D]]): state trajectories
            prior_reg (float): prior regularization

        Returns:
            None
        """
        # create variables
        results = {'losses': [], 'success': False, 'num_iters': 1}
        I = np.identity(self.num_dofs * self.total_num_basis)  # shape: DMxDM

        # compute initial loss
        prev_loss = self.compute_loss(Y, kind='marginal_log_likelihood')
        results['losses'].append(prev_loss)

        weights = []
        # for each trajectory in Y
        for y in Y:
            # compute phase
            T = len(y)
            phases = np.linspace(0, 1, T)

            # compute weight vector (shape: DM)
            y = y.reshape(-1)  # shape: 2DT
            Phi = np.hstack([self.Phi(s) for s in phases])  # shape: DMx2DT

            weight = (np.linalg.inv(Phi.dot(Phi.T) + prior_reg * I)).dot(Phi.dot(y))  # shape: DM

            weights.append(weight)

        weights = np.array(weights)  # shape: NxDM

        # fit a gaussian and set weight distribution
        mean = Gaussian.compute_mean(weights)  # shape: DM
        cov = Gaussian.compute_covariance((weights - mean), bessels_correction=False)  # shape: DMxDM

        self._weights = Gaussian(mean=mean, covariance=cov)

        # compute loss
        loss = self.compute_loss(Y, kind='marginal_log_likelihood')
        results['losses'].append(loss)
        results['success'] = loss < prev_loss

        return results

    def imitate(self, Y, prior_reg=1., method='lrr', num_iters=1000, threshold=1e-4, verbose=False):
        r"""
        Imitate given trajectories, i.e. learn the parameters :math:`\theta = \{\mu_w, \Sigma_w\}` of the Gaussian
        distribution put on the weights.

        Note that because we are trying to model the distribution, multiple demonstrations need to be given.

        Args:
            Y (np.array[N,T,2D], list[np.array[T,2D]]): state trajectories
            method (str): method to use when learning ('lrr' for linear ridge regression, 'em' for expectation-
                maximization, 'mml' for maximum_marginal_likelihood)
            prior_reg (float): prior regularization
            num_iters (int): number of iterations for the EM algo
            threshold (float): convergence threshold for the EM algo
            verbose (bool): if we should print details during the optimization process

        Returns:
            None
        """
        # linear ridge regression
        if method == 'lrr':
            return self.linear_ridge_regression(Y, prior_reg=prior_reg)

        # expectation-maximization
        elif method == 'em':
            return self.expectation_maximization(Y, num_iters=num_iters, threshold=threshold, verbose=verbose)

        # maximum marginal likelihood
        elif method == 'mml':
            return self.maximum_marginal_likelihood(Y, prior_reg=prior_reg, num_iters=num_iters, threshold=threshold,
                                                    verbose=verbose)
        else:
            raise NotImplementedError("The specified method has not been implemented")

    # alias
    fit = imitate

    def condition(self, s, y_desired_mean, y_desired_covariance):
        r"""
        Modulate the trajectory distribution by conditioning.

        .. math:: p(w | y^*)

        Args:
            s (float): phase value
            y_desired_mean (np.array[2D]): desired output state
            y_desired_covariance (np.array[2D]): desired output covariance. If it's low (thus the precision is high)
                it means, that we want to reach that point

        Returns:

        """
        # # construct joint distribution between new point and weight
        # w_mean, w_cov = self.weights.mean, self.weights.cov     # shape: DM and DMxDM
        # Phi_s = self.Phi(s)     # shape: DMx2D
        # joint_mean = np.concatenate((w_mean, Phi_s.T.dot(w_mean)))
        # joint_cov = np.vstack((np.hstack((w_cov, w_cov.dot(Phi_s))),
        #                       np.hstack((Phi_s.T.dot(w_cov), Phi_s.T.dot(w_cov).dot(Phi_s) + y_desired_covariance))))
        # gaussian = Gaussian(mean=joint_mean, covariance=joint_cov)
        # # compute posterior mean by conditioning the joint Gaussian
        # weight = gaussian.condition(y_desired_mean, range(w_mean.size))

        # construct the joint distribution between the new point (i.e. mean) and weight, and then compute the
        # posterior on the weight by conditioning the joint distribution given the new desired mean and covariance
        return self.posterior_weights(y_desired_mean, s, y_cov=y_desired_covariance)

    # alias
    # modulate = condition

    def combine(self, mps, priorities):
        r"""
        Modulate the trajectory distribution by combining/co-activating different movement primitives. This returns
        a ProMP which represents the combination of the given ProMPs.

        .. math:: p(\tau) = \prod_{i=1}^P p_i(\tau)^{\alpha_i}

        where :math:`\alpha_i \in [0,1]` are the priorities.

        Notes: this is a specific case of the `blend()` method, where the same priority is set for all the phase
        values.

        Args:
            mps (list[ProMP]): list of ProMPs
            priorities (list[float]): list of priorities with the same length as the number of given `mps`

        Returns:
            ProMP: resulting ProMP
        """
        return self.blend(mps=mps, priorities=priorities)

    def blend(self, mps, priorities):
        r"""
        Modulate the trajectory distribution by blending different movement primitives together. This returns
        a ProMP which represents the combination of the given ProMPs.

        .. math::

            p(\tau) = \prod{t=1}^T p(y_t)
            p(y_t) = \prod_{i=1}^P p_i(y_t)^{\alpha_{i,t}}
            p_i(y_t) = \int p_i(y_t | w_i) p_i(w_i) dw_i

        Warnings: note that the above formula assumes independence instead of conditional independence. Indeed,
        :math:`p(\tau | w) = \prod_{t=1}^T p(y_t | w)`, and :math:`p(\tau; \theta) = \int p(\tau | w) p(w; \theta)
        dw = \int \prod_{t=1}^T p(y_t | w) p(w) dw` which is different from :math:`p(\tau; \theta) = \prod_{t=1}^T
        p(y_t; \theta) = \prod_{t=1}^T \int p(y_t| w) p(w; \theta) dw`. Note the product symbol which has switched
        with the integral symbol. The former expression assumes conditional independence between the :math:`y_t`
        given :math:`w`, while the latter assumes that the :math:`y_t` are completely independent between each
        other, which is a stronger assumption.

        Args:
            mps (list[ProMP]): list of ProMPs
            priorities (np.array[K,T]): list of priorities, where each priority is an array of float numbers
                representing the priority / activation factor for each phase value.

        Returns:
            ProMP: resulting ProMP
        """
        # check that the number of priorities and the number of ProMP coincide
        if len(mps) != len(priorities):
            raise ValueError("The number of priorities is different from the number of ProMPs")

        # set for each ProMP its priority
        for mp, priority in zip(mps, priorities):
            mp.priority = priority

        # return the combination
        return ProMP(promps=mps)

    # alias
    # TODO def sequence
    # sequence = blend

    def power(self, priority):
        """
        Set the priority (aka activation function) which represents the degree of activation of this ProMP.

        Args:
            priority (float, int, callable): priority function
        """
        self.priority = priority

    def multiply(self, other):
        r"""
        Multiply a ProMP with another ProMP.

        The resulting prediction (in the case for the marginal prediction) will be given by:

        .. math:: p^*(y_t) = p_1(y_t; \theta_1)^{\alpha_1} p_2(y_t; \theta_2)^{\alpha_2}

        where the :math:`\alpha` are the priorities set beforehand.

        Notes: multiplying a ProMP by a square matrix (to rotate or scale the prediction) can be done by first
        calling `predict_proba()` or `step_proba()` which return a Gaussian distribution, and then multiply this
        last one by the matrix. The same can be carried out to add a vector or to perform an affine transformation.

        Args:
            other (ProMP): other ProMP

        Returns:
            ProMP: resulting ProMP
        """
        # if other == ProMP
        if isinstance(other, ProMP):
            if self.promps and other.promps:
                return ProMP(promps=self.promps + other.promps)
            elif self.promps:
                return ProMP(promps=self.promps + [other])
            elif other.promps:
                return ProMP(promps=[self] + other.promps)
            else:
                return ProMP(promps=[self, other])
        else:
            raise TypeError("Trying to multiply a ProMP with {}, which has not be defined".format(type(other)))

    #############
    # Operators #
    #############

    def __str__(self):
        """Return description of this class"""
        return self.__class__.__name__

    def __call__(self, s, probabilistic=True, method='marginal', return_gaussian=True, sample=False):
        """Predict output given the phase"""
        if probabilistic:
            return self.predict_proba(s, method=method, return_gaussian=return_gaussian)
        return self.predict(s, method=method, sample=sample)

    def __len__(self):
        """Return the number of degree of freedoms if one ProMP. Else, return the number of ProMPs"""
        if self.promps:
            return len(self.promps)
        return self.num_dofs

    def __getitem__(self, idx):
        """
        Return the specified probabilistic movement primitive(s)

        Args:
            idx (int, slice): index

        Returns:
            ProMP: the interested ProMPs
        """
        # if multiple ProMPs, return the one specified
        if self.promps:
            return self.promps[idx]

        # check number of movement primitives
        if isinstance(idx, int):
            num_dofs = 1
        elif isinstance(idx, slice): # slice
            num_dofs = abs(idx.stop - idx.start) / abs(idx.step)
        else:
            raise TypeError("Expecting the given index to be an integer or a slice")

        # create probabilistic movement primitive of the same type
        if isinstance(self, DiscreteProMP):
            promp = DiscreteProMP(num_dofs=num_dofs, weights=self.weights[idx])
        elif isinstance(self, RhythmicProMP):
            promp = RhythmicProMP(num_dofs=num_dofs, weights=self.weights[idx])
        else:  # ProMP
            promp = ProMP(num_dofs=num_dofs, weights=self.weights[idx])

        # set block diagonal basis matrix
        promp.Phi = self.Phi[idx]

        # return ProMP
        return promp

    def __iter__(self):
        """Iterate over the probabilistic movement primitives"""
        for i in range(len(self)):
            yield self[i]

    def __pow__(self, priority):
        """Set the priority"""
        self.power(priority=priority)

    def __mul__(self, other):
        """Multiply this ProMP with another one"""
        return self.multiply(other)

    def __rmul__(self, other):
        """Multiply this ProMP with another one"""
        return self.multiply(other)


class DiscreteProMP(ProMP):
    r"""Discrete ProMP

    ProMP to be used for discrete / stroke-based movements.
    """
    def __init__(self, num_dofs, num_basis, weights=None, canonical_system=None, noise_covariance=1.,
                 basis_width=None):
        """
        Initialize the Discrete ProMP.

        Args:
            num_dofs (int): number of degrees of freedom (denoted by `D`)
            num_basis (int): number of basis functions (denoted by `M`)
            weights (np.array[DM], Gaussian, None): the weights that can be optimized. If None, it will create a
                custom weight array.
            canonical_system (CS, None): canonical system. If None, it will create a Linear canonical system that goes
                from `t0=0` to `tf=1`.
            noise_covariance (np.array[2D,2D]): covariance noise matrix
            basis_width (None, float): width of the basis. By default, it will be 1./(2*num_basis) such that the
                basis_width represents the standard deviation, and such that 2*std_dev = 1./num_basis.
        """
        super(DiscreteProMP, self).__init__(num_dofs=num_dofs, weight_size=num_dofs*num_basis, weights=weights,
                                            canonical_system=canonical_system, noise_covariance=noise_covariance)

        # define the basis width if not defined
        if basis_width is None:
            basis_width = 1./(2*num_basis)

        # create Gaussian basis matrix with shape: DMx2D
        if num_dofs == 1:
            self.Phi = GaussianBM(self.cs, num_basis, basis_width=basis_width)
        else:
            self.Phi = BlockDiagonalMatrix([GaussianBM(self.cs, num_basis, basis_width=basis_width)
                                            for _ in range(num_dofs)])


class RhythmicProMP(ProMP):
    r"""Rhythmic ProMP

    ProMP to be used for rhythmic movements.
    """

    def __init__(self, num_dofs, num_basis, weights=None, canonical_system=None, noise_covariance=1.,
                 basis_width=None):
        """
        Initialize the Rhythmic ProMP.

        Args:
            num_dofs (int): number of degrees of freedom (denoted by `D`)
            num_basis (int): number of basis functions (denoted by `M`)
            weights (np.array[DM], Gaussian, None): the weights that can be optimized. If None, it will create a
                custom weight array.
            canonical_system (CS, None): canonical system. If None, it will create a Linear canonical system that goes
                from `t0=0` to `tf=1`.
            noise_covariance (np.array[2D,2D]): covariance noise matrix
            basis_width (None, float): width of the basis. By default, it will be 1./(2*num_basis) such that the
                basis_width represents the standard deviation, and such that 2*std_dev = 1./num_basis.
        """
        super(RhythmicProMP, self).__init__(num_dofs=num_dofs, weight_size=num_dofs * num_basis, weights=weights,
                                            canonical_system=canonical_system, noise_covariance=noise_covariance)

        # define the basis width if not defined
        if basis_width is None:
            basis_width = 1. / (2 * num_basis)

        # create Von-Mises basis matrix with shape: DMx2D
        if num_dofs == 1:
            self.Phi = VonMisesBM(self.cs, num_basis, basis_width=basis_width)
        else:
            self.Phi = BlockDiagonalMatrix([VonMisesBM(self.cs, num_basis, basis_width=basis_width)
                                            for _ in range(num_dofs)])


# TESTS
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def plot_state(Y, title=None, linewidth=1.):
        y, dy = Y.T
        plt.figure()
        if title is not None:
            plt.suptitle(title)

        # plot position y(t)
        plt.subplot(1, 2, 1)
        plt.title('y(t)')
        plt.plot(y, linewidth=linewidth)  # TxN

        # plot velocity dy(t)
        plt.subplot(1, 2, 2)
        plt.title('dy(t)')
        plt.plot(dy, linewidth=linewidth)  # TxN


    def plot_weighted_basis(promp):
        phi_track = promp.weighted_basis(t)  # shape: DM,T,2D

        plt.subplot(1, 2, 1)
        plt.plot(phi_track[:, :, 0].T, linewidth=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(phi_track[:, :, 1].T, linewidth=0.5)


    # create data and plot it
    N = 8
    t = np.linspace(0., 1., 100)
    eps = 0.1
    y = np.array([np.sin(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])      # shape: NxT
    dy = np.array([2*np.pi*np.cos(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])     # shape: NxT
    Y = np.dstack((y, dy))  # N,T,2D  --> why not N,2D,T
    plot_state(Y, title='Training data')
    plt.show()

    # create discrete and rhythmic ProMP
    promp = DiscreteProMP(num_dofs=1, num_basis=10, basis_width=1./20)

    # plot the basis function activations
    plt.plot(promp.Phi(t)[:, :, 0].T)
    plt.title('basis functions')
    plt.show()

    # plot ProMPs
    y_pred = promp.rollout()
    plot_state(y_pred[None], title='ProMP prediction before learning', linewidth=2.)    # shape: N,T,2D
    plot_weighted_basis(promp)
    plt.show()

    # learn from demonstrations
    promp.imitate(Y)
    y_pred = promp.rollout()
    plot_state(y_pred[None], title='ProMP prediction after learning', linewidth=2.)   # N,T,2D
    plot_weighted_basis(promp)
    plt.show()

    # modulation: final positions (goals)

    # modulation: final velocities

    # modulation: via-points

    # combination/co-activation/superposition

    # blending
