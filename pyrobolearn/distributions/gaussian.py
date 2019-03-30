#!/usr/bin/env python
"""Define the Multivariate Gaussian / Normal distribution class.

This distribution is so important in the field of Machine Learning that we extended the functionalities of the basic
`torch.distributions.MultivariateNormal`. This distribution will notably be used for Gaussian Mixture Models,
Probabilistic Movement Primitives, Kernelized Movement Primitives, etc.

References:
    [1] torch.distributions: https://pytorch.org/docs/stable/distributions.html
    [2] Gaussian distribution: pyrobolearn/models/gaussian
"""

import torch
import numpy as np
import scipy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Gaussian(torch.distributions.MultivariateNormal):
    r"""Multivariate Gaussian distribution (aka normal distribution)

    The multivariate Gaussian distribution also known as the multivariate normal distribution is the most well-known
    and probably the most used distribution because of its nice mathematical properties, its appearance in different
    arguments/theorems (such as the maximum entropy argument, and central limit theorem), as well as its different
    extensions (Gaussian mixture models, Gaussian processes, and so on).

    The multivariate Gaussian distribution is given by:

    .. math:: p(x) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
                        \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)

    where :math:`x` is the real random vector, :math:`\mu` is the mean, and :math:`\Sigma` is the covariance matrix.

    The Gaussian distribution has multiple interesting mathematical properties, notably [1]:
    * The average of random variables tend to a Gaussian distribution by the central limit theorem
    * Given the two first moments (i.e. mean and covariance), it is the maximum entropy distribution.
    * The sum of two independent Gaussian random variables (with the same dimension) is also Gaussian. This
        is the same as saying that the convolution of two Gaussian PDFs is a Gaussian PDF.
    * The product of two Gaussians is also Gaussian but it is no more a valid probability distribution.
    * The affine transformation of a Gaussian variable is again Gaussian.
    * The conditional distribution is also Gaussian.
    * The marginal distribution of a multivariate Gaussian with respect to a subset of the variables is itself Gaussian.
    * A kernel matrix (which compares the similarity between different samples) can be provided instead of a
        covariance matrix (which compares how the different dimensionalities vary between each other).

    Note that given the first and second moment (mean and covariance respectively), the Gaussian distribution is
    the maximum entropy distribution. That is, it is the distribution that maximizes the entropy (which makes thus the
    least number of assumptions about the data). As a side note, if only the first moment is provided, the one that
    maximized the entropy is the Gibbs distribution, and if no moments are provided, then it is the uniform
    distribution. This argument is known as the maximum entropy argument. Because the first and second moments can be
    quite reliably estimated from the data, the Gaussian distribution is often used in the ML field.

    Note that the conjugate prior for the mean of the Gaussian distribution is also Gaussian. As for the covariance
    matrix, its conjugate prior is the inverse Wishart distribution.

    Note that because the covariance is a symmetric, positive semi-definite matrix, it has a Cholesky decomposition.
    Thus, it can be expressed as the product of a lower triangular matrix with its transpose :math:`\Sigma = LL^\top`.
    This is useful for two reasons:
    - any lower triangular matrices multiplied with its transpose results in a symmetric positive semi-definite
     matrix, which thus represents a proper covariance matrix. This can be for instance useful when predicting a
     full covariance matrix with a neural network. Indeed, it is hard to enforce that type of constraint (i.e. making
     sure that the produced covariance matrix is symmetric and positive semi-definite) while optimizing the network.
     We can thus instead output a lower-triangular matrix and multiplied by its transpose.
    - it allows to solve efficiently a system of linear equations :math:`Ax = b` without having to compute the inverse
     (and thus, the determinant). This is achieved in a 2-step way, by first computing :math:`Ly=b` for :math:`y` by
     forward substitution, and then computing :math:`L^\top x = y` by backward substitution.

    References:
        [1] "Pattern Recognition and Machine Learning", Bishop, 2006
        [2] "Machine Learning: A Probabilistic Perspective", Murphy, 2012, chap 3 and 4
        [3] "The Matrix Cookbook", Petersen et al., 2012, sec 8

    The implementation of this class was inspired by the following codes:
    * `scipy`: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
    * `autograd`:  https://github.com/HIPS/autograd/blob/master/autograd/scipy/stats/multivariate_normal.py
    * `torch`: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html
    * `riepybdlib`: https://gitlab.martijnzeestraten.nl/martijn/riepybdlib/blob/master/riepybdlib/statistics.py
    * `pbdlib`: https://gitlab.idiap.ch/rli/pbdlib-python/blob/master/pbdlib/mvn.py
    """

    # Set array priority (matrices have __array_priority__ equal to 10.0)
    # This is useful when multiplying a matrix with a Gaussian object.
    # For more info, see:
    # - https://docs.scipy.org/doc/numpy/reference/ufuncs.html
    # - https://stackoverflow.com/questions/38229953/array-and-rmul-operator-in-python-numpy
    __array_priority__ = 1002

    def __init__(self, mean, covariance=None, precision=None, tril=None, seed=None, N=None):  # manifold=None,
        """
        Initialize the multivariate normal distribution on the given manifold.

        Args:
            mean (torch.Tensor): mean of the distribution
            covariance (torch.Tensor, None): positive-definite covariance matrix
            precision (torch.Tensor, None): positive-definite precision matrix
            tril (torch.Tensor, None): lower-triangular factor of covariance, with positive-valued diagonal
            seed (int): random seed. Useful when sampling.
            manifold (None): By default, it is the Euclidean space.
            N (int, N): the number of data points
        """
        # call superclass
        super(Gaussian, self).__init__(mean, covariance_matrix=covariance, precision_matrix=precision,
                                       scale_tril=tril)

        # set the seed
        self.seed = seed

        # TODO: formulas are different depending on the space we are in
        # TODO: the manifold should be an object (see the `geomstats` module)
        # For now, it will just be a string and we will focus on the Euclidean space
        self.manifold = 'euclidean'  # manifold

        # number of data points
        self.N = N

    ##############
    # Properties #
    ##############

    # aliases
    @property
    def covariance(self):
        return self.covariance_matrix

    @property
    def precision(self):
        return self.precision_matrix

    @property
    def tril(self):
        return self.scale_tril

    @property
    def mode(self):
        """Return the mode; the value that is the most likely to be sampled"""
        return self.mean

    @property
    def dim(self):
        """dimensionality of the gaussian distribution"""
        return self.mean.size(-1)

    @property
    def normalization_constant(self):
        """normalization constant"""
        if self.covariance is not None:
            return self.compute_normalization_constant(self.covariance)

    @property
    def seed(self):
        """Return the seed"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed"""
        if seed is not None:
            torch.manual_seed(seed)
        self._seed = seed

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_parametric():
        """The Gaussian distribution is a nonparametric model; the mean and covariance summarized the data"""
        return False

    @staticmethod
    def is_linear():
        """The Gaussian doesn't have parameters. Even if the mean and covariance are considered as parameters,
        the model is not linear wrt them"""
        return False

    @staticmethod
    def is_recurrent():
        """The Gaussian is not recurrent model; current outputs do not depend on previous inputs"""
        return False

    @staticmethod
    def is_probabilistic():
        """The Gaussian distribution is by definition a probabilistic model"""
        return True

    @staticmethod
    def is_discriminative():
        """The Gaussian is not a discriminative model; no inputs are involved"""
        return False

    @staticmethod
    def is_generative():
        """The Gaussian is a generative model, and thus we can sample from it"""
        return True

    @staticmethod
    def compute_mean(X, dim=0):
        r"""
        Compute the empirical mean vector given the data. This is also known as the maximum likelihood estimate for
        the mean vector :math:`\mu`, i.e. :math:`\max_{\mu} p(X | \mu, \Sigma)`.

        Args:
            X (torch.Tensor[N,D]): data matrix of shape NxD (if axis=0) or DxN (if axis=1), where N is the number of
                samples, and D is the dimensionality of a data point
            dim (int): axis along which the mean is computed

        Returns:
            float[D]: mean vector
        """
        # if manifold is Euclidean
        return torch.mean(X, dim=dim)

    @staticmethod
    def compute_covariance(X, axis=0, bessels_correction=True):
        r"""
        Compute the empirical covariance matrix given the data. This is also known as the maximum likelihood estimate
        for the covariance matrix :math:`\Sigma`, i.e. :math:`\max_{\Sigma} p(X | \mu, \Sigma)`.

        Args:
            X (array[N,D]): data matrix of shape NxD where N is the number of samples, and D is the dimensionality
                of a data point
            axis (int): axis along which the covariance is computed
            bessels_correction (bool): if True, it will compute the covariance using `1/N-1` instead of `N`.

        Returns:
            float[D,D]: 2D covariance matrix
        """
        # if manifold is Euclidean
        cov = np.cov(X.numpy(), rowvar=bool(axis), bias=not bessels_correction)
        return torch.from_numpy(cov).float()

    @staticmethod
    def compute_precision(X, axis=0, bessels_correction=True):
        r"""
        Compute the empirical precision matrix given the data.

        Args:
            X (array[N,D]): data matrix of shape NxD where N is the number of samples, and D is the dimensionality
                of a data point
            axis (int): axis along which the precision is computed
            bessels_correction (bool): if True, it will compute the precision using `1/N-1` instead of `N`.

        Returns:
            float[D,D]: 2D precision matrix
        """
        prec = torch.inverse(Gaussian.compute_covariance(X, axis=axis, bessels_correction=bessels_correction))
        return prec

    @staticmethod
    def compute_normalization_constant(covariance):
        r"""
        Compute the normalization constant based on the covariance, which is given by:

        .. math:: c = \frac{1}{(2\pi)^{\frac{d}{2}} |\Sigma|^{\frac{1}{2}}}

        Args:
            covariance (array_like: float[d,d]): covariance matrix

        Returns:
            float: normalization constant such that the distribution sums to 1 when integrated.
        """
        size = covariance.shape[0]
        normalization_constant = 1. / ((2 * np.pi) ** (size / 2.) * torch.det(covariance) ** 0.5)
        return normalization_constant

    @staticmethod
    def is_symmetric(X, tol=1e-8):
        """Check if given matrix X is symmetric.
        If a matrix is symmetric, it has real eigenvalues, orthogonal eigenvectors and is always diagonalizable.
        """
        return torch.allclose(X, X.t(), atol=tol)

    # TODO: check if X belongs to the SPD space S^n_{++}
    @staticmethod
    def is_psd(X, tol=1e-12):
        """Check if given matrix is PSD"""
        evals, evecs = torch.eig(X, eigenvectors=False)
        return torch.all(evals >= 0 - tol)

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Returns an iterator over the model parameters."""
        yield self.mean
        yield self.covariance

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        yield "mean", self.mean
        yield "covariance", self.covariance

    def pdf(self, value):  # likelihood
        r"""
        Probability density function evaluated at the given `x`.

        This is given by the following formula:

        .. math:: p(x) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
                            \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)

        where :math:`\Sigma` is the covariance matrix, :math:`\mu` is the mean, and :math:`d` is the dimensionality
        of the Gaussian distribution.

        Args:
            value (torch.Tensor): vector to evaluate the probability density function.

        Returns:
            float: probability density evaluated at `x`.
        """
        return torch.exp(self.log_prob(value))

    def logcdf(self, value):
        r"""
        Log of the Cumulative Distribution Function.

        Note that this requires at least scipy v1.1.0

        Args:
            value (torch.Tensor): vector to evaluate the log of the cumulative distribution function.

        Returns:
            float: log of the cumulative distribution function evaluated at `x`.
        """
        return torch.log(self.cdf(value))

    def distance(self, x):
        r"""
        Compute the distance of the given data from the mean by also taking into account the covariance. In the
        'Euclidean' space, this method returns the Mahalanobis distance which is defined as
        :math:`D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}`.

        Args:
            x (torch.Tensor[D]): data vector

        Returns:
            float: distance
        """
        if self.manifold == 'euclidean':
            diff = x - self.mean
            return torch.sqrt(diff.mm(self.precision).mm(diff))

    def condition(self, input_value, output_idx, input_idx=None):
        r"""
        Compute the conditional distribution.

        Assume the joint distribution :math:`p(x_1, x2)` is modeled as a normal distribution, then
        the conditional distribution of :math:`x_1` given :math:`x_2` is given by
        :math:`p(x_1|x_2) = \mathcal{N}(\mu, \Sigma)` (which is also Gaussian), where the mean :math:`\mu` and
        covariance :math:`\Sigma` are given by:

        .. math::

            \mu &= \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2) \\
            \Sigma &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}

        Args:
            input_value (float[d2]): array of values :math:`x_2` such that we have :math:`p(x_1|x_2)`
            output_idx (int[d1], int): indices that we are interested in, given (i.e. conditioned on) the other ones.
                That is, the indices for :math:`x_1`
            input_idx (int[d2], int, None): indices that we conditioned on, i.e. corresponding to the values. If None,
                it will be inferred. That is, the indices for :math:`x_2`

        Returns:
            Gaussian: Conditional Normal distribution
        """
        # aliases
        value, o, i = input_value, output_idx, input_idx

        value = torch.Tensor([value]) if isinstance(value, (int, float)) else torch.tensor(value)

        # if i is None:
        #     o = torch.Tensor([o]) if isinstance(o, int) else torch.Tensor(o)
        #     # from all the indices remove the output indices
        #     i = torch.Tensor(list(set(range(self.dim)) - set(o)))
        #     i.sort()
        #
        #     # make sure that the input indices have the same length as the value ones
        #     i = i[:len(value)]
        # else:
        #     i = torch.Tensor([i]) if isinstance(i, int) else torch.Tensor(i)
        # assert len(i) == len(value), "The value array and the idx2 array have different lengths"

        if i is None:
            o = np.array([o]) if isinstance(o, int) else np.array(o)
            # from all the indices remove the output indices
            i = np.array(list(set(range(self.dim)) - set(o)))
            i.sort()

            # make sure that the input indices have the same length as the value ones
            i = i[:len(value)]
        else:
            i = np.array([i]) if isinstance(i, int) else np.array(i)
        assert len(i) == len(value), "The value array and the idx2 array have different lengths"

        # compute conditional
        c = self.covariance[np.ix_(o, i)].mm(torch.inverse(self.covariance[np.ix_(i, i)]))
        mu = self.mean[o] + c.mm((value - self.mean[i]).view(-1, 1))[:, 0]
        cov = self.covariance[i, o]
        if len(cov.size()):
            cov = cov.view(-1, 1)
        cov = self.covariance[np.ix_(o, o)] - c.mm(cov)
        return Gaussian(mean=mu, covariance=cov)

    def marginalize(self, idx):
        r"""
        Compute and return the marginal distribution (which is also Gaussian) of the specified indices.

        Let's assume that the joint distribution :math:`p(x_1, x_2)` is modeled as a Gaussian distribution, that is:

        .. math:: x \sim \mathcal{N}(\mu, \Sigma)

        where :math:`x = [x_1, x_2]`, :math:`\mu = [\mu_1, \mu_2]` and
        :math:`\Sigma=\left[\begin{array}{cc} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{array}\right]`

        then the marginal distribution :math:`p(x_1) = \int_{x_2} p(x_1, x_2) dx_2` is also Gaussian and is given by:

        .. math:: p(x_1) = \mathcal{N}(\mu_1, \Sigma_{11})

        Args:
            idx (int, slice): indices of :math:`x_1` (this value should be between 0 and D-1, where D is
                the dimensionality of the data)

        Returns:
            Gaussian: marginal distribution (which is also Gaussian)
        """
        if isinstance(idx, (int, float)):
            idx = [idx]
        return Gaussian(mean=self.mean[idx], covariance=self.covariance[np.ix_(idx, idx)])

    def multiply(self, other):
        r"""
        Multiply a Gaussian by another Gaussian, by a square matrix (under an affine transformation), or a float
        number.

        The product of two Gaussian PDFs is given by:

        .. math:: \mathcal{N}(\mu_1, \Sigma_1) \mathcal{N}(\mu_2, \Sigma_2) = C \mathcal{N}(\mu, \Sigma)

        where :math:`C = \mathcal{N}(\mu_1; \mu_2, \Sigma_1 + \Sigma_2)` is a constant (scalar),
        :math:`\Sigma = (\Sigma_1^{-1} + \Sigma_2^{-1})^-1`, and
        :math:`\mu = \Sigma (\Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2)`.

        Note that the product of two Gaussians is a Gaussian, but it is usually no more a valid probability density
        function. In order to make it a proper probability distribution, we have to normalize it, which results to
        remove the constant :math:`C`.

        The product of a Gaussian distribution :math:`\mathcal{N}(\mu, \Sigma)` with a square matrix :math:`A` gives:

        .. math:: Ax \sim \mathcal{N}(A \mu, A \Sigma A^T)

        The product of a Gaussian by a float does nothing as we have to re-normalize it to be a proper distribution.

        Args:
            other (Gaussian, array_like of float[D,D], float): Gaussian, square matrix (to rotate or scale), or float

        Returns:
            Gaussian: resulting Gaussian distribution
        """
        # if other == Gaussian
        if isinstance(other, Gaussian):
            # coefficient = Gaussian(other.mean, self.covariance + other.covariance)(self.mean) * self.coefficient
            prec1, prec2 = self.precision, other.precision
            cov = torch.inverse(prec1 + prec2)

            size = self.mean.size()
            if len(size) == 1:
                mean1 = self.mean.view(-1, 1)
                mean2 = other.mean.view(-1, 1)
            else:
                mean1 = self.mean
                mean2 = other.mean
    
            mu = (cov.mm(prec1.mm(mean1) + prec2.mm(mean2))).view(size)
            return Gaussian(mean=mu, covariance=cov)  # , coefficient=coefficient)

        # if other == square matrix
        elif isinstance(other, torch.Tensor):
            size = self.mean.size()
            if len(size) == 1:
                mean = self.mean.view(-1, 1)
            else:
                mean = self.mean
            return Gaussian(mean=(other.mm(mean)).view(size), covariance=other.mm(self.covariance).mm(other.t()))

        # if other == number
        elif isinstance(other, (int, float)):
            return self

        else:
            raise TypeError("Trying to multiply a Gaussian with {}, which has not be defined".format(type(other)))

    def get_multiplication_coefficient(self, other):
        r"""
        Return the coefficient :math:`C` that appears when multiplying two Gaussians.

        As a reminder, the product of two Gaussian PDFs is given by:

        .. math:: \mathcal{N}(\mu_1, \Sigma_1) \mathcal{N}(\mu_2, \Sigma_2) = C \mathcal{N}(\mu, \Sigma)

        where :math:`C = \mathcal{N}(\mu_1; \mu_2, \Sigma_1 + \Sigma_2)` is a constant (scalar),
        :math:`\Sigma = (\Sigma_1^{-1} + \Sigma_2^{-1})^-1`, and
        :math:`\mu = \Sigma (\Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2)`.

        Args:
            other (Gaussian): other Gaussian

        Returns:
            float: resulting coefficient
        """
        return Gaussian(mean=other.mean, covariance=self.covariance + other.covariance)(self.mean)

    # def integrate_conjugate_prior(self, other): # TODO: call it marginal_likelihood
    def marginal_distribution(self, x, prior):
        r"""
        Integrate the given Gaussian conjugate prior on the parameters with the current Gaussian PDF.

        .. math::

            p(y; \theta) &= \int \mathcal{N}(y | \Phi(x) w, \Sigma_y) \mathcal{N}(w | \mu_w, \Sigma_w) dw \\
                         &= \mathcal{N}(y | \Phi(x) \mu_w, \Phi(x) \Sigma_w \Phi(x)^T + \Sigma_y)

        Args:
            prior (Gaussian): the other Gaussian conjugate prior

        Returns:
            Gaussian: resulting Gaussian
        """
        if isinstance(prior, Gaussian):
            if callable(self.mean):
                # TODO works with functions instead of arrays
                Phi_x = self.mean.grad(x, self.mean.parameters)
                # TODO check when the mean and covariance don't have the same shape
                return Phi_x * Gaussian(prior.mean, prior.covariance) + Gaussian(0, self.covariance)
        else:
            raise TypeError("Expecting the prior to be a Gaussian distribution on the parameters of the mean "
                            "of this Gaussian")

    def power(self, exponent):
        r"""
        Raise to the power this Gaussian.

        Note that raising a Gaussian distribution to the power 2 is the same as multiplying the Gaussian with itself.

        Warnings: if the exponent is small, the resulting covariance will be large. Inversely, if the exponent
        is very large the resulting covariance will be small. If the exponent is zero, this results in a uniform
        distribution.

        Args:
            exponent (float): strictly positive number

        Returns:
            Gaussian: resulting gaussian distribution
        """
        if exponent <= 0:
            raise ValueError("The exponent has to be a strictly positive number")
        return Gaussian(mean=self.mean, covariance=1. / exponent * self.covariance)

    def add(self, other):
        r"""
        The sum of independent Gaussian random variables (with the same dimension) is also Gaussian. This can
        also be used to sum a Gaussian with a vector (affine transformation).

        The sum of two independent Gaussian RVs (with the same dimensionality), such that
        :math:`x_1 \sim \mathcal{N}(\mu_1, \Sigma_1)` and :math:`x_2 \sim \mathcal{N}(\mu_2, \Sigma_2)`, is given
        by :math:`x_1 + x_2 \sim \mathcal{N}(\mu_1 + \mu_2, \Sigma_1 + \Sigma_2)`

        The sum of a Gaussian distribution :math:`x \sim \mathcal{N}(\mu, \Sigma)` with a vector :math:`v` results in
        a translation of this distribution, given by :math:`x \sim \mathcal{N}(\mu + v, \Sigma)`.

        Args:
            other (Gaussian, float[d]): the other Gaussian distribution, or a vector.

        Returns:
            Gaussian: resulting sum of two independent Gaussian distribution, or resulting sum of a Gaussian with
                a vector.
        """
        if isinstance(other, Gaussian):
            return Gaussian(mean=self.mean + other.mean, covariance=self.covariance + other.covariance)
        return Gaussian(mean=self.mean + other, covariance=self.covariance)

    def affine_transform(self, A, b=None):
        r"""
        Perform an affine transformation on the Gaussian PDF. If :math:`x \sim \mathcal{N}(\mu, \Sigma)`, then
        :math:`Ax+b ~ \mathcal{N}(A\mu + b, A^T \Sigma A)`.

        Args:
            A (torch.Tensor[D,D]): square matrix
            b (torch.Tensor[D]): vector

        Returns:
            Gaussian: resulting Gaussian PDF
        """
        size = self.mean.size()
        if len(size) == 1:
            mean = self.mean.view(-1, 1)
        else:
            mean = self.mean

        if b is None:
            return Gaussian(mean=A.mm(mean).view(size), covariance=A.mm(self.covariance).mm(A.t()))
        return Gaussian(mean=A.mm(mean).view(size) + b, covariance=A.mm(self.covariance).mm(A.t()))

    def integrate(self, lower=None, upper=None):
        r"""
        Integrate the gaussian distribution between the two given bounds.

        Args:
            lower (torch.Tensor[D], float, None): lower bound (default: -np.inf)
            upper (torch.Tensor[D], float, None): upper bound (default: np.inf)

        Returns:
            float: p(lower <= x <= upper)
        """
        mean_shape = tuple(self.mean.shape)

        # check lower bound
        if lower is None:
            lower = np.full(mean_shape, -np.inf)
        elif isinstance(lower, float):
            lower = np.full(mean_shape, lower)

        # check upper bound
        if upper is None:
            upper = np.full(mean_shape, np.inf)
        elif isinstance(upper, float):
            upper = np.full(mean_shape, upper)

        # integrate
        return scipy.stats.mvn.mvnun(lower, upper, self.mean.numpy(), self.covariance.numpy())[0]

    def grad(self, x, wrt='x'):
        r"""
        Compute the gradient of the Gaussian distribution evaluated at the given data. Let's
        :math:`p(x; \mu, \Sigma) = \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
        \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)` be the multivariate Gaussian distribution.

        Then (using [1]), we have:

        .. math::

            \frac{\partial p(x; \mu, \Sigma)}{\partial x} &= - p(x) \Lambda (x - \mu) \\
            \frac{\partial p(x; \mu, \Sigma)}{\partial \mu} &= p(x) \Lambda (x - \mu) \\
            \frac{\partial p(x; \mu, \Sigma)}{\partial \Sigma} &= \frac{1}{2} p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda
                - \Lambda) \\
            \frac{\partial p(x; \mu, \Lambda)}{\partial \Lambda} &= \frac{1}{2} p(x) (\Sigma - (x-\mu)(x-\mu)^T)

        where :math:`\Lambda = \Sigma^{-1}` is the precision matrix.

        Args:
            x (torch.Tensor[D]): data vector
            wrt (str): specify with respect to which variable we compute the gradient. It can take the following
                values 'x', 'mu' or 'mean', 'sigma' or 'covariance', 'lambda' or 'precision'.

        Returns:
            torch.Tensor: gradient of the same shape (as the variable from which we take the gradient)

        References:
            [1] "The Matrix Cookbook" (math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), Petersen and Pedersen, 2012
        """
        # TODO: check when x is a matrix
        wrt = wrt.lower()
        if wrt == 'x':
            return - self.pdf(x) * self.precision.mm(x - self.mean)
        elif wrt == 'mu' or wrt == 'mean':
            return self.pdf(x) * self.precision.mm(x - self.mean)
        elif wrt == 'sigma' or wrt[:3] == 'cov':
            mu, L = self.mean, self.precision
            diff = x - mu
            return 0.5 * self.pdf(x) * (L.mm(torch.ger(diff, diff)).mm(L) - L)
        elif wrt == 'lambda' or wrt == 'precision':
            diff = x - self.mean
            return 0.5 * self.pdf(x) * (self.covariance - torch.ger(diff, diff))
        else:
            raise ValueError("The given 'wrt' argument is not valid (see documentation)")

    def grad_fn(self, wrt='x'):
        r"""
        Compute the gradient function of the Gaussian wrt the current parameters (mean and covariance). Let's
        :math:`p(x; \mu, \Sigma) = \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
        \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)` be the multivariate Gaussian distribution.

        Then (using [1]), we have:

        .. math::

            \frac{\partial p(x; \mu, \Sigma)}{\partial x} &= - p(x) \Lambda (x - \mu) \\
            \frac{\partial p(x; \mu, \Sigma)}{\partial \mu} &= p(x) \Lambda (x - \mu) \\
            \frac{\partial p(x; \mu, \Sigma)}{\partial \Sigma} &= \frac{1}{2} p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda
                - \Lambda)
            \frac{\partial p(x; \mu, \Lambda)}{\partial \Lambda} &= \frac{1}{2} p(x) (\Sigma - (x-\mu)(x-\mu)^T)

        where :math:`\Lambda = \Sigma^{-1}` is the precision matrix.

        Args:
            wrt (str): specify with respect to which variable we compute the gradient. It can take the following
                values 'x', 'mu' or 'mean', 'sigma' or 'covariance', 'lambda' or 'precision'.

        Returns:
            callable: gradient function that can be evaluated later

        References:
            [1] "The Matrix Cookbook" (math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), Petersen and Pedersen, 2012
        """
        # TODO: check when x is a matrix
        wrt = wrt.lower()
        if wrt == 'x':
            def wrap(pdf, mu, L):
                def grad(x):
                    return - pdf(x) * L.mm(x - mu)

                return grad
        elif wrt == 'mu' or wrt == 'mean':
            def wrap(pdf, mu, L):
                def grad(x):
                    return pdf(x) * L.mm(x - mu)

                return grad
        elif wrt == 'sigma' or wrt[:3] == 'cov':
            def wrap(pdf, mu, L):
                def grad(x):
                    diff = x - mu
                    return 0.5 * pdf(x) * (L.mm(torch.ger(diff, diff)).mm(L) - L)

                return grad
        elif wrt == 'lambda' or wrt == 'precision':
            def wrap(pdf, mu, S):
                def grad(x):
                    diff = x - mu
                    return 0.5 * pdf(x) * (S - torch.ger(diff, diff))

                return grad

            return wrap(self.pdf, self.mean, self.covariance)
        else:
            raise ValueError("The given 'wrt' argument is not valid (see documentation)")
        return wrap(self.pdf, self.mean, self.precision)

    def hessian(self, x, wrt='x'):
        r"""
        Compute the Hessian matrix (2nd derivative) of the Gaussian distribution evaluated at the given data. Let's
        :math:`p(x; \mu, \Sigma) = \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
        \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)` be the multivariate Gaussian distribution.

        Then (using [1]), we have:

        .. math::

            \frac{\partial^2 p(x; \mu, \Sigma)}{\partial x^2} &= p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda - \Lambda) \\
            \frac{\partial^2 p(x; \mu, \Sigma)}{\partial \mu^2} &= p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda - \Lambda)

        where :math:`\Lambda = \Sigma^{-1}` is the precision matrix.

        Args:
            x (torch.Tensor[D]): data vector
            wrt (str): specify with respect to which variable we compute the gradient. It can take the following
                values 'x', 'mu' or 'mean'.

        References:
            [1] "The Matrix Cookbook" (math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), Petersen and Pedersen, 2012
        """
        # TODO: check when x is a matrix
        # TODO: derive the hessian wrt the covariance and precision matrix
        if wrt == 'x' or wrt == 'mu' or wrt == 'mean':
            mu, L = self.mean, self.precision
            diff = x - mu
            return self.pdf(x) * (L.mm(torch.ger(diff, diff)).mm(L) - L)
        else:
            raise ValueError("The given 'wrt' argument is not valid (see documentation)")

    def hessian_fn(self, wrt='x'):
        r"""
        Compute the Hessian function of the Gaussian wrt to the current parameters (mean and covariance). Let's
        :math:`p(x; \mu, \Sigma) = \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^\frac{d}{2} |\Sigma|^\frac{1}{2}}
        \exp\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)` be the multivariate Gaussian distribution.

        Then (using [1]), we have:

        .. math::

            \frac{\partial^2 p(x; \mu, \Sigma)}{\partial x^2} &= p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda - \Lambda) \\
            \frac{\partial^2 p(x; \mu, \Sigma)}{\partial \mu^2} &= p(x) (\Lambda (x-\mu)(x-\mu)^T \Lambda - \Lambda)

        where :math:`\Lambda = \Sigma^{-1}` is the precision matrix.

        Args:
            wrt (str): specify with respect to which variable we compute the gradient. It can take the following
                values 'x', 'mu' or 'mean'.

        References:
            [1] "The Matrix Cookbook" (math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), Petersen and Pedersen, 2012
        """
        # TODO: check when x is a matrix
        # TODO: derive the hessian wrt the covariance and precision matrix
        if wrt == 'x' or wrt == 'mu' or wrt == 'mean':
            def wrap(pdf, mu, L):
                def grad(x):
                    diff = x - mu
                    return self.pdf(x) * (L.mm(torch.ger(diff, diff)).mm(L) - L)

                return grad
        else:
            raise ValueError("The given 'wrt' argument is not valid (see documentation)")
        return wrap(self.pdf, self.mean, self.precision)

    def update(self, x):
        r"""
        Update the Gaussian distribution by taking into account the new given point(s) `x`. Specifically, it updates
        the mean and covariance matrix based on the given point(s) `x`. This is only valid if the initial data was
        provided, as we need to know the total number of data points that were previously given.

        The updated mean is given by:

        .. math:: \mu_{N + M} = \frac{N}{N + M} \mu_N + \frac{M}{N + M} \mu_M

        where :math:`N` is the initial number of data points, :math:`M` is the number of given data points,
        :math:`\mu_N` is the previous mean, and :math:`\mu_M` is the mean computed on the given data.

        The udpated covariance (without the bessels correction) is given by:

        .. math::

            \Sigma_{N+M} = \frac{N}{N+M} (\Sigma_N + \mu_N\mu_N^T) + \frac{M}{N+M} (\Sigma_M + \mu_M\mu_M^T)
                            - \mu_{N+M}\mu_{N+M}^T

        Args:
            x (torch.Tensor): data vector/matrix
        """
        if self.N is None:
            # if the mean and covariance are not defined, learn from scratch
            if self.mean is None and self.covariance is None:
                self.mle(x)
            else:
                raise RuntimeError("The number of data points was never specified. This is needed in order to "
                                   "update the mean and covariance in an online fashion")
        else:
            # compute the number of data points
            N = self.N
            M = 1 if len(x.shape) == 1 else x.shape[0]

            # compute the updated mean
            mu_N, mu_M = self.mean, self.compute_mean(x)
            ratio_N, ratio_M = float(N) / (N + M), float(M) / (N + M)
            mu = ratio_N * mu_N + ratio_M * mu_M

            # compute the updated covariance
            # TODO: check math with bessel correction
            sigma_N, sigma_M = self.covariance, self.compute_covariance(x)
            cov = ratio_N * (sigma_N + torch.ger(mu_N, mu_N)) + ratio_M * (sigma_M + torch.ger(mu_M, mu_M)) - \
                  torch.ger(mu, mu)

            # update the mean, covariance, and number of data points
            super(Gaussian, self).__init__(loc=mu, covariance_matrix=cov)
            self.N = N + M

    def mle(self, data):
        r"""
        Perform maximum likelihood estimate (MLE) given the data. This results to compute the empirical mean and
        covariance from the data.

        .. math:: \max_\theta p(X | \theta)

        where :math:`\theta` is the set of parameters, which in this case are the mean and covariance matrix,
        i.e. :math:`\theta = \{ \mu, \Sigma \}`, and :math:`X` represents the data set.

        Args:
            data (torch.Tensor[N,D]): data matrix of shape NxD

        Returns:
            float: value of the maximum likelihood estimate obtained
        """
        mean = self.compute_mean(data)
        covariance = self.compute_covariance(data)
        super(Gaussian, self).__init__(loc=mean, covariance_matrix=covariance)
        self.N = data.shape[0]
        return self.pdf(data)

    # alias
    fit = mle
    maximum_likelihood = mle

    def ellipsoid_axes(self):  # ellipse_confidence # confidence_region
        r"""
        Compute the axes of the ellipsoid defined by the covariance matrix. Specifically, it returns the main
        axes of the ellipsoid (i.e. its orientation) as well as their scaling.

        Returns:
            torch.Tensor[D]: square root of eigenvalues in descending order
            torch.Tensor[D,D]: eigenvectors arranged in column vectors in descending order
        """
        evals, evecs = np.linalg.eigh(self.covariance.numpy())
        return torch.from_numpy(np.sqrt(evals[::-1]).copy()).float(), torch.from_numpy(evecs[:, ::-1].copy()).float()

    # def __array_ufunc__(self, *args):
    #     print(args)

    #############
    # Operators #
    #############

    def __len__(self):
        """dimensionality of the Gaussian distribution"""
        return self.dim

    def __call__(self, x=None, size=None):
        """
        If no arguments are given, it returns one sample from the distribution. If a vector is provided, it returns
        the probability associated with this one (i.e. how probable it is that the given sample was generated from
        this Gaussian distribution), that is the pdf evaluated at the given vector.

        Args:
            x (torch.Tensor, None): vector to evaluate the probability density function.
            size (int, None): number of samples

        Returns:
            float, or torch.Tensor: probability density evaluated at `x`, or samples
        """
        if x is not None:
            return self.pdf(x)
        return self.sample_n(n=size)

    def __getitem__(self, idx):
        """
        Conditional and marginal distribution.

        Args:
            idx (int, slice, tuple): if int or slice, it will return the marginal distribution. If tuple, it
                will return the conditional distribution.

        Returns:
            Gaussian: conditional or marginal distribution

        Examples:
            # joint distribution p(x1,x2)
            g = Gaussian(torch.Tensor([1.,2.]), np.identity(2))

            # marginal distribution p(x1) and p(x2)
            marg1 = g[0]
            marg2 = g[1]

            # conditional distribution p(x1|x2) and p(x2|x1)
            sample = g.sample()
            cond1 = g[0,1,sample[0]]
            cond2 = g[1,0,sample[1]]
        """
        if isinstance(idx, tuple):  # conditional distribution
            if len(idx) == 2:
                value, idx1 = idx
                idx2 = None
            elif len(idx) == 3:
                value, idx1, idx2 = idx
            else:
                raise IndexError("Expecting two or three indices: value, idx1 (, idx2)")
            return self.condition(value, idx1, idx2)
        else:  # marginal distribution
            return self.marginalize(idx)

    def __add__(self, other):
        """
        The sum of two independent Gaussian random variables (with the same dimension) is also Gaussian. This can
        also be used to sum a Gaussian with a vector (affine transformation).

        Args:
            other (Gaussian, float[d]): the other Gaussian distribution, or a vector.

        Returns:
            Gaussian: resulting sum of two independent Gaussian distribution, or resulting sum of a Gaussian with
                a vector.
        """
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        """
        Multiply two Gaussian distributions, or multiply a gaussian distribution with a matrix (as done during
        an affine transformation).

        Args:
            other (Gaussian, array_like of float[D,D]): Gaussian, or square matrix (to rotate or scale)

        Returns:
            Gaussian: resulting Gaussian distribution
        """
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __pow__(self, exponent):
        """Apply the power exponent on the Gaussian"""
        return self.power(exponent)


######################
# Plotting functions #
######################


def plot_3d(ax, X, Y, pdf, title=None, xlabel='x1', ylabel='x2', zlabel='x3'):
    if isinstance(pdf, (tuple, list)):
        pdf = np.max(np.dstack(pdf), axis=-1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    ax.plot_surface(X, Y, pdf, cmap='viridis', linewidth=0)


def plot_2d_contour(ax, x, y, pdf, title=None, xlabel='x1', ylabel='x2'):
    if isinstance(pdf, (tuple, list)):
        pdf = np.max(np.dstack(pdf), axis=-1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.contourf(x, y, pdf)


def plot_3d_and_2d_countour(gaussians, step=500, bound=10, fig=None, title='', block=True):
    if not isinstance(gaussians, (list, tuple)):
        gaussians = [gaussians]

    # Create grid and multivariate normal
    x = torch.linspace(-bound, bound, step)
    y = torch.linspace(-bound, bound, step)
    X, Y = torch.meshgrid(x, y)
    pos = torch.zeros(X.shape + (2,))
    pos[:, :, 0], pos[:, :, 1] = X, Y
    pdf = [gaussian.pdf(pos) for gaussian in gaussians]

    # convert back to numpy
    x = x.numpy()
    y = y.numpy()
    X, Y = X.numpy(), Y.numpy()
    pdf = [p.numpy() for p in pdf]

    # if more than one gaussian, fuse by taking the maximum
    if len(pdf) > 1:
        pdf = np.max(np.dstack(pdf), axis=-1)
    else:
        pdf = pdf[0]

    # create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))  # Twice as wide as it is tall.
    plt.suptitle(title)

    # 1st subplot (3D)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3d(ax, X, Y, pdf, title='p(x1, x2)', xlabel='x1', ylabel='x2', zlabel='p')

    # 2nd subplot (2D)
    ax = fig.add_subplot(1, 2, 2)
    plot_2d_contour(ax, x, y, pdf, title='p(x1, x2)', xlabel='x1', ylabel='x2')

    # show plot
    fig.tight_layout()
    plt.show(block=block)


def plot_2d_ellipse(ax, gaussian, color='g', fill=False, plot_2devs=False, plot_arrows=True):
    # alias
    g = gaussian

    # compute std deviation and eigenvectors from the gaussian
    std_dev, evecs = g.ellipsoid_axes()
    std_dev, evecs = std_dev.numpy(), evecs.numpy()

    # plot ellipse from standard deviations and eigenvectors
    width, height = 2 * std_dev[0], 2 * std_dev[1]
    angle = np.rad2deg(np.arccos(evecs[:, 0].dot([1, 0])))
    # if 3rd or 4th quadrant, we need to reverse the sign for the angle
    x, y = evecs[:, 0]
    if y < 0:
        angle = -angle

    facecolor = color if fill else 'none'

    # 2 ellipses (one std dev and two std dev)
    ellipse_2std = Ellipse(xy=g.mean, width=2 * width, height=2 * height, angle=angle, edgecolor=color, lw=2,
                           facecolor=facecolor, alpha=0.5)
    ax.add_artist(ellipse_2std)
    if plot_2devs:
        ellipse_1std = Ellipse(xy=g.mean, width=width, height=height, angle=angle, edgecolor=color, lw=2,
                               facecolor=facecolor, alpha=0.8)
        ax.add_artist(ellipse_1std)

    # if we need to plot arrows
    if plot_arrows:
        # compute scaled eigenvectors
        p = std_dev * evecs
        # draw arrows
        ax.arrow(g.mean[0], g.mean[1], p[0, 0], p[1, 0], length_includes_head=True, head_width=0.15, color='r')
        ax.arrow(g.mean[0], g.mean[1], p[0, 1], p[1, 1], length_includes_head=True, head_width=0.15, color='r')

    return ellipse_2std


def plot_3d_and_2d_conditional(joint_gaussian, cond_gaussian, x1_value=0., step=500, bound=10, block=True):

    # Create grid and multivariate normal
    x = np.linspace(-bound, bound, step)
    y = np.linspace(-bound, bound, step)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0], pos[:, :, 1] = X, Y
    joint_pdf = joint_gaussian.pdf(torch.from_numpy(pos).float()).numpy()
    cond_pdf = cond_gaussian.pdf(torch.from_numpy(x).float().view(-1, 1)).numpy()
    z_max = pdf.max()

    # create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))  # Twice as wide as it is tall.
    plt.suptitle('Conditional Gaussian')

    # 1st subplot (3D)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3d(ax, X, Y, joint_pdf, title='p(x1,x2)', xlabel='x1', ylabel='x2', zlabel='p')

    # draw plane that cut the gaussian
    y1 = np.linspace(-bound, bound, 2)
    z = np.linspace(0, z_max, 2)
    Y1, Z = np.meshgrid(y1, z)
    ax.plot_surface(x1_value, Y1, Z, color='red', alpha=0.4)
    # cset = ax.contourf(X, Y, pdf, zdir='x', offset=-bound, cmap=cm.coolwarm)

    # 2nd subplot (2D)
    ax = fig.add_subplot(1, 2, 2)
    # ax.plot(x, pdf[step / 2 + x1_value * step / (2 * bound)])
    ax.plot(x, cond_pdf)
    ax.set(title='p(x2|x1)', xlabel='x2', ylabel='p(x2|x1)')

    # show plot
    fig.tight_layout()
    plt.show(block=block)


# TESTS
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Ellipse

    # create two 2D Gaussian distributions
    m1, c1 = torch.tensor([0., 0.]), torch.eye(2) * 0.5
    m2, c2 = torch.tensor([1.5, 1.5]), torch.tensor([[1., 0.5], [0.5, 2.]])
    g1 = Gaussian(m1, c1)
    g2 = Gaussian(m2, c2)

    # sample from the Gaussian distributions, and plot them
    d1 = g1.sample((200,)).numpy()
    d2 = g2.sample((200,)).numpy()
    fig, ax = plt.subplots(1, 1)
    ax.set(title='sampling from 2 Gaussians', aspect='equal')
    ax.scatter(d1[:, 0], d1[:, 1], color='b', alpha=0.7)
    ax.scatter(d2[:, 0], d2[:, 1], color='r', alpha=0.7)
    plt.show()

    # 3D and 2D plots of the Gaussian distributions
    plot_3d_and_2d_countour([g1, g2], title='joint distribution')

    ##################
    # Use 1 Gaussian #
    ##################

    # check if the Gaussian distribution produces the same results as `scipy.stats.multivariate_normal`
    from scipy.stats import multivariate_normal

    # create grid and multivariate normal
    bound, step = 10., 500
    x = np.linspace(-bound, bound, step)
    y = np.linspace(-bound, bound, step)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0], pos[:, :, 1] = X, Y
    # evaluate PDF using Gaussian and Scipy
    pdf = g2.pdf(torch.from_numpy(pos).float()).numpy()
    pdf2 = multivariate_normal(g2.mean.numpy(), g2.covariance.numpy()).pdf(pos)
    print("Same PDF produced by Gaussian and Scipy: {}".format(np.allclose(pdf, pdf2)))

    # check if valid PDF: integrate from -inf to inf the Gaussian distribution (it should be equal to 1)
    # \int \int p(x1,x2) dx1 dx2 = 1
    # p(x1, x2) = pdf; dx = 2.*bound/step; dx1 dx2 = (2.*bound/step)**2
    print("Integration from -inf to inf: {}".format(g2.integrate()))
    print("Summation: {}".format((2. * bound / step) ** 2 * pdf.sum()))

    # samples from the Gaussian and plot ellipse
    samples = g2.sample((100,)).numpy()
    fig, ax = plt.subplots(1, 1)
    ax.set(title='Sampling from one Gaussian', aspect='equal')
    ax.scatter(samples[:, 0], samples[:, 1], color='b')
    plot_2d_ellipse(ax, g2, fill=True, plot_2devs=True, plot_arrows=True)
    plt.show()

    # conditional distribution of the Gaussian p(y|x)
    x_value = 0.
    g_cond = g2.condition(input_value=x_value, output_idx=1)
    plot_3d_and_2d_conditional(g2, g_cond, x_value)

    # marginalization of the Gaussian by summing and using the normal distribution
    # by summing
    dx = 2. * bound / step
    y_sum = pdf.sum(axis=0) * dx
    # using gaussian marginalize function
    g_margin = g2.marginalize(idx=0)
    y_margin = g_margin.pdf(torch.from_numpy(x).float().view(-1, 1)).numpy()
    # plot
    fig, axes = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    axes[0].plot(x, y_sum, color='blue')
    axes[0].set(title='Marginalization by integrating', xlabel='x2', ylabel='p(x2)')
    axes[1].plot(x, y_margin, color='red')
    axes[1].set(title='Marginalization using indexing', xlabel='x2', ylabel='p(x2)')
    fig.tight_layout()
    plt.show()

    # affine transformation on the Gaussian distribution
    b = torch.tensor([-2., -1.])
    theta = torch.tensor(np.deg2rad(-45))
    A = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]])
    # g_aff = A * g2 + b
    g_aff = g2 * A + b
    print("Gaussian under affine transformation: mean={} and cov={}".format(g_aff.mean.numpy(),
                                                                            g_aff.covariance.numpy()))
    samples = g_aff.sample((500,)).numpy()
    fig, ax = plt.subplots(1, 1)
    ax.set(title='Gaussian under affine transformation', aspect='equal')
    ax.scatter(samples[:, 0], samples[:, 1], color='b')
    plot_2d_ellipse(ax, g_aff)
    plt.show()

    ###################
    # use 2 Gaussians #
    ###################

    # addition of two independent Gaussians
    g_sum = g1 + g2
    fig, ax = plt.subplots(1, 1)
    ax.set(title='addition', xlim=[-5, 5], ylim=[-5, 5], aspect='equal')
    e1 = plot_2d_ellipse(ax, g1, color='g', plot_arrows=False)
    e2 = plot_2d_ellipse(ax, g2, color='b', plot_arrows=False)
    e3 = plot_2d_ellipse(ax, g_sum, color='r', plot_arrows=False)
    ax.legend([e1, e2, e3], ['G1', 'G2', 'G1+G2'], loc=2)
    plt.show()

    # multiplication of two independent Gaussians
    g_mul = g1 * g2
    fig, ax = plt.subplots(1, 1)
    ax.set(title='multiplication', xlim=[-5, 5], ylim=[-5, 5], aspect='equal')
    e1 = plot_2d_ellipse(ax, g1, color='g', plot_arrows=False)
    e2 = plot_2d_ellipse(ax, g2, color='b', plot_arrows=False)
    e3 = plot_2d_ellipse(ax, g_mul, color='r', plot_arrows=False)
    ax.legend([e1, e2, e3], ['G1', 'G2', 'G1*G2'], loc=2)
    plt.show()

    ################################
    # Fit a Gaussian on given data #
    ################################

    # create data
    g_data = Gaussian(mean=torch.tensor([2., 3.]), covariance=torch.tensor([[1, -0.5], [-0.5, 1]]))
    print(g_data.mean)
    print(g_data.covariance)
    samples = np.random.multivariate_normal(mean=g_data.mean.numpy(), cov=g_data.covariance.numpy(), size=1000)

    # fit one Gaussian and plot it along the data
    g = Gaussian(mean=torch.zeros(2), covariance=torch.eye(2))
    g.fit(torch.from_numpy(samples).float())
    plot_3d_and_2d_countour(g_data, title='Gaussian that generated the data', block=False)
    plot_3d_and_2d_countour(g, title='fitted Gaussian')
