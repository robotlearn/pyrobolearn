#!/usr/bin/env python
"""Provide the various common probability distribution layers / modules.

This file provides layers / modules that can output probability distributions that inherit from
`torch.distributions.*`. Several pieces of code were inspired from [1, 2].

References:
    [1] torch.distributions: https://pytorch.org/docs/stable/distributions.html
    [2] pytorch-a2c-ppo-acktr:
        - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py
        - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/distributions.py
    [3] Gaussian distribution: pyrobolearn/models/gaussian
"""

from abc import ABCMeta

import collections
import numpy as np
import torch

from pyrobolearn.distributions.categorical import Categorical as CategoricalDistribution
from pyrobolearn.distributions.bernoulli import Bernoulli as BernoulliDistribution
from pyrobolearn.distributions.gaussian import Gaussian as GaussianDistribution
from pyrobolearn.distributions.gmm import GMM as GaussianMixtureDistribution


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "Ilya Kostrikov"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def init_module(module, init_weight=None, init_bias=None):
    """
    Initialize the given module using the given initialization scheme for the weight and bias terms.

    The user can select the initialization scheme from `torch.nn.init`. This function is taken from [1] and modified.

    Args:
        module (torch.nn.Module): torch module to initialize.
        init_weight (callable, None): this is a callable function that accepts as input the module to initialize its
            weights. If None, it will leave the module weights untouched.
        init_bias (callable, None): this is a callable function that accepts as input the module to initialize its
            bias terms. If None, it will leave the module bias untouched.

    Returns:
        torch.nn.Module: initialized module.

    References:
        [1] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py
    """
    if init_weight is not None:
        init_weight(module.weight.data)
    if init_bias is not None:
        init_bias(module.bias.data)
    return module


def wrap_init_tensor(init_tensor, *args, **kwargs):
    r"""Define a higher order function that accepts as inputs the initial method to initialize a tensor, as well
    as its arguments, and returns a function that only awaits for its tensor input. With this, you can use the above
    :func:`init_module` function quite easily. For instance:

    Examples:
        >>> module = torch.nn.Linear(in_features=10, out_features=5)
        >>> weight_init = wrap_init_tensor(torch.nn.init.orthogonal_, gain=1.)
        >>> weight_bias = wrap_init_tensor(torch.nn.init.constant_, val=0)
        >>> module = init_module(module, wrap_init_tensor(module, weight_init, weight_bias))

    Returns:
        callable: return the callable function that only accepts a tensor as input.
    """
    def init(tensor):
        return init_tensor(tensor, *args, **kwargs)
    return init


def init_orthogonal_weights_and_constant_biases(module, gain=1., val=0.):
    """Initialize the weights of the module to be orthogonal and with a bias of 0s. This is inspired by [1].

    Args:
        gain (float): optional scaling factor for the orthogonal weights.
        val (float): val: the value to fill the bias tensor with.

    Returns:
        torch.nn.Module: the initialized module.

    References:
        [1] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
    """
    weight_init = wrap_init_tensor(torch.nn.init.orthogonal_, gain=gain)
    weight_bias = wrap_init_tensor(torch.nn.init.constant_, val=val)
    module = init_module(module, weight_init, weight_bias)
    return module


class DistributionModule(torch.nn.Module):
    r"""Probability distribution module

    Define a wrapper around `torch.distributions.Distribution` with lazy creation and additional features.
    Everytime this object is called given an input it wraps that given input, and return a distribution over it.
    """

    def __init__(self, distribution):
        """
        Initialize the distribution.

        Args:
            distribution (type): subclass of `torch.distributions.Distribution`.
        """
        super(DistributionModule, self).__init__()
        self.distribution = distribution

    @property
    def distribution(self):
        """Return the distribution type."""
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        """Set the distribution."""
        if not issubclass(distribution, torch.distributions.Distribution):
            raise TypeError("Expecting the given distribution to be a subclass of `torch.distributions.Distribution`, "
                            "instead got: {}".format(distribution))
        self._distribution = distribution

    def forward(self, x):
        """Forward the given inputs :attr:`x`."""
        raise NotImplementedError


class FixedVectorModule(torch.nn.Module):
    r"""Fixed vector generator (module)

    Generate the fixed vector. This generates a fixed vector everytime it is called.
    If N samples are given at the input such that it has a shape (N,I), it returns N copy of the vector of shape (N,M).
    """

    def __init__(self, vector):
        """
        Initialize the fixed generator.

        Args:
            vector (torch.Tensor): fixed vector
        """
        super(FixedVectorModule, self).__init__()
        if not isinstance(vector, torch.Tensor):
            raise TypeError("Expecting the vector to be an instance of `torch.Tensor`, instead got: "
                            "{}".format(type(vector)))
        self._vector = vector

    def forward(self, x):
        """Take as input the base output vector / matrix and return the diagonal covariance matrix(ces).

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: vector / matrix of shape (N, D)
        """
        if len(x.shape) == 1:
            return self._vector
        return self._vector.repeat(x.size(0), 1)


class VectorModule(torch.nn.Module):
    r"""Vector generator (module)

    Generate an output vector / matrix given an input vector / matrix. Specifically, it is just a linear module.
    """

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize the mean generator.

        Args:
            num_inputs (int): size of the base output vector
            num_outputs (int): size of the output vector
        """
        super(VectorModule, self).__init__()
        # linear mapping between the base output vector / matrix and the mean output vector / matrix
        model = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self._model = init_orthogonal_weights_and_constant_biases(model)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the vector / matrix.

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: vector / matrix of shape (N, D)
        """
        return self._model(x)


class IdentityModule(torch.nn.Module):
    r"""Identity Module.

    This just returns whatever tensor is given.
    """
    def __init__(self):
        """Initialize the identity module."""
        super(IdentityModule, self).__init__()

    def forward(self, x):
        """Return the same given input :attr:`x`."""
        return x


class FixedMeanModule(torch.nn.Module):  # this is the same as FixedVector (but with different documentation)
    r"""Fixed mean generator

    Generate the mean of the multivariate Normal distribution. This generates a fixed mean everytime it is called.
    If N samples are given at the input such that it has a shape (N,I), it returns N copy of the mean of shape (N,M).
    """

    def __init__(self, mean):
        """
        Initialize the mean generator.

        Args:
            mean (torch.Tensor): mean vector
        """
        super(FixedMeanModule, self).__init__()
        if not isinstance(mean, torch.Tensor):
            raise TypeError("Expecting the mean vector to be an instance of `torch.Tensor`, instead got: "
                            "{}".format(type(mean)))
        self._mean = mean

    def forward(self, x):
        """Take as input the base output vector / matrix and return the mean vector / matrix.

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: mean vector / matrix of shape (N, D)
        """
        if len(x.shape) == 1:
            return self._mean
        return self._mean.repeat(x.size(0), 1)


class MeanModule(torch.nn.Module):  # this is the same as VectorModule (but with different documentation)
    r"""Mean generator

    Generate the mean of the multivariate Normal distribution.
    """

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize the mean generator.

        Args:
            num_inputs (int): size of the base output vector
            num_outputs (int): size of the action mean vector
        """
        super(MeanModule, self).__init__()
        # linear mapping between the base output vector / matrix and the mean output vector / matrix
        model = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self._model = init_orthogonal_weights_and_constant_biases(model)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the mean vector / matrix.

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: mean vector / matrix of shape (N, D)
        """
        return self._model(x)


class FixedStandardDeviationModule(torch.nn.Module):
    r"""Fixed Standard Deviation Module

    Generate the standard deviations of the Normal distributions. This generates a fixed standard deviations everytime
    it is called.
    If N samples are given at the input such that it has a shape (N,I), it returns N copy of the mean of shape (N,M).
    """

    def __init__(self, stddev=None, variances=None):
        """
        Initialize the fixed standard deviation generator.

        Args:
            stddev (torch.Tensor, None): vector of standard deviations. If the variances is None, the standard
                deviations will be considered.
            variances (torch.Tensor, None): vector of variances. If None, the standard deviations have to be defined.
        """
        super(FixedStandardDeviationModule, self).__init__()
        if stddev is None:
            if variances is None:
                raise ValueError("The variances or the standard deviations have to be specified")
            # check if negative elements in variances
            if torch.any(variances < 0.):
                raise ValueError("Expecting the variances to be strictly positive, found some negative variances: "
                                 "{}".format(type(variances)))
            stddev = variances.sqrt()

        # check if negative elements in standard deviations
        if torch.any(stddev < 0.):
            raise ValueError("Expecting the standard deviations to be strictly positive, found some negative "
                             "deviations: {}".format(type(stddev)))

        # if standard deviations are very close to zero
        if torch.any(stddev.isclose(torch.tensor(0.))):
            # add small offset
            stddev += 1.e-4

        self._stddev = stddev

    def forward(self, x):
        """Take as input the base output vector / matrix and return the standard deviation vector / matrix.

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: standard deviation vector / matrix of shape (N, D)
        """
        if len(x.shape) == 1:
            return self._stddev
        return self._stddev.repeat(x.size(0), 1)


class StandardDeviationModule(torch.nn.Module):
    r"""Standard Deviation Module

    Generate the standard deviations of the Normal distribution.
    """

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize the mean generator.

        Args:
            num_inputs (int): size of the base output vector
            num_outputs (int): size of the action mean vector
        """
        super(StandardDeviationModule, self).__init__()
        # linear mapping between the base output vector / matrix and the mean output vector / matrix
        model = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self._model = init_orthogonal_weights_and_constant_biases(model)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the standard deviation vector / matrix.

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: standard deviation vector / matrix of shape (N, D)
        """
        # compute standard deviations
        x = self._model(x)  # shape (N,D) or (D,)

        # get standard deviation by taking the exponential (which is always positive)
        x = torch.exp(x)  # shape (N,D) or (D,)

        # return the standard deviation
        return x


class FixedDiagonalCovarianceModule(torch.nn.Module):
    r"""Fixed Diagonal Covariance Generator

    This covariance receives the mean as input, and generates samples using that mean and a fixed diagonal covariance
    matrix set during the instantiation of this class.

    Note that the standard deviations must be non-negatives.
    """

    def __init__(self, variances=None, stddev=None):
        """
        Initialize the fixed diagonal covariance matrix generator.

        Args:
            variances (torch.Tensor, None): vector of variances. If None, the standard deviations have to be defined.
            stddev (torch.Tensor, None): vector of standard deviations. If the variances is None, the standard
                deviations will be considered.
        """
        super(FixedDiagonalCovarianceModule, self).__init__()
        if variances is None:
            if stddev is None:
                raise ValueError("The variances or the standard deviations have to be specified")
            variances = stddev.pow(2)

        # check if negative elements in variances
        if torch.any(variances < 0.):
            raise ValueError("Expecting the variances to be strictly positive, found some negative variances: "
                             "{}".format(type(variances)))

        # if variance very close to zero
        if torch.any(variances.isclose(torch.tensor(0.))):
            # add small offset
            variances += 1.e-4

        # create fixed diagonal covariance matrix
        dim = variances.size(-1)
        self._covariance = torch.diag(variances).view(1, dim, dim)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the fixed diagonal covariance matrix(ces).

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: covariance matrices (one covariance matrix for each sample) of shape (N, D, D)
        """
        # return the fixed diagonal covariance
        if len(x.shape) == 1:
            return self._covariance
        # stack N times the covariance
        return self._covariance.repeat(x.size(0), 1, 1)


class FixedCovarianceModule(torch.nn.Module):
    r"""Fixed Covariance Generator

    This Gaussian receives the mean as input, and generates samples using that mean and a fixed covariance matrix set
    during the instantiation of this class.
    """

    def __init__(self, covariance=None, precision=None, tril=None):
        """
        Initialize the fixed covariance matrix generator.

        Args:
            covariance (None, torch.Tensor): positive-definite covariance matrix.
            precision (None, torch.Tensor): positive-definite precision matrix.
            tril (None, torch.Tensor): lower triangular matrix which is the Cholesky decomposition of the covariance
                matrix, with positive-valued diagonal.
        """
        super(FixedCovarianceModule, self).__init__()
        if covariance is None and precision is None and tril is None:
            raise ValueError("The covariance, the precision, or the lower triangular factor of the covariance has to "
                             "be specified.")
        if covariance is not None:
            dim = covariance.size(-1)
        if precision is not None:
            dim = precision.size(-1)
        if tril is not None:
            dim = tril.size(-1)

        # let the PyTorch framework check if the covariance matrix is correct
        distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=covariance,
                                                              precision_matrix=precision, scale_tril=None)

        # get the covariance matrix
        self._covariance = distribution.covariance_matrix

    def forward(self, x):
        """Take as input the base output vector / matrix and return the fixed covariance matrix(ces).

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: covariance matrices (one covariance matrix for each sample) of shape (N, D, D)
        """
        if len(x.shape) == 1:
            return self._covariance(x)
        # stack N times the covariance
        return self._covariance.repeat(x.size(0), 1, 1)


class DiagonalCovarianceModule(torch.nn.Module):
    r"""Diagonal Covariance Generator

    This class receives generates the diagonal of a covariance matrix given the base output vector / matrix.
    Note that a covariance matrix has to be positive semi-definite. In the case of a diagonal matrix,
    this is achieved by having the diagonal entries (=the variances) to be non-negative. If the standard deviations
    are given they can have any values as we will square them later to form the diagonal entries of the covariance
    matrix. By squaring them, they all become non-negative. If instead the variance are given as inputs we have to
    make sure that they are non-negative. A quick hack is to give the logarithm of the variance which is always
    positive.
    """

    def __init__(self, num_inputs, num_outputs, offset=1.e-4):
        """
        Initialize the diagonal covariance generator.

        Args:
            num_inputs (int): size of the base output vector
            num_outputs (int): size of the action mean vector
            offset (float): small offset to be added to the diagonal elements of the covariance matrix such that
                it is positive definite.
        """
        super(DiagonalCovarianceModule, self).__init__()
        # set output dimension
        self._dim = int(num_outputs)

        # define diagonal offset such that the covariance is positive definite
        self._offset = offset * torch.diag(torch.ones(self._dim))

        # linear mapping between the base output vector / matrix to the covariance diagonal elements
        model = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self._model = init_orthogonal_weights_and_constant_biases(model)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the diagonal covariance matrix(ces).

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: covariance matrices (one covariance matrix for each sample) of shape (N, D, D)
        """
        # from base outputs to vector representing the log var
        x = self._model(x)  # shape (N,D) or (D,)

        # get variance by taking the exponential (which is always positive)
        x = torch.exp(x)  # shape (N,D) or (D,)

        # create covariance matrix of shape (N,D,D) or (D,D)
        if len(x.shape) > 1:
            covariance = torch.zeros(x.size(0), self._dim, self._dim)  # shape (N,D,D)

            # set the elements in the covariance matrix  # TODO: remove this for-loop
            for i in range(len(covariance)):
                covariance[i] = torch.diag(x[i]) + self._offset
        else:
            covariance = torch.diag(x) + self._offset  # shape (D,D)

        # return the diagonal covariance matrix
        return covariance


class FullCovarianceModule(torch.nn.Module):
    r"""Full Covariance Generator

    This class generates a full covariance matrix given the base output vector / matrix, and maps it to the lower
    triangular matrix of the Cholesky decomposition of the covariance matrix, which is then multiplied by its
    transpose to give back the full covariance matrix.

    A quick reminder:

    Because the covariance is a symmetric, positive semi-definite matrix, it has a Cholesky decomposition. Thus, it
    can be expressed as the product of a lower triangular matrix with its transpose :math:`\Sigma = LL^\top`.
    This is useful for two reasons:
    - any lower triangular matrices multiplied with its transpose results in a symmetric positive semi-definite
     matrix, which thus represents a proper covariance matrix. This can be for instance useful when predicting a
     full covariance matrix with a neural network. Indeed, it is hard to enforce that type of constraint (i.e. making
     sure that the produced covariance matrix is symmetric and positive semi-definite) while optimizing the network.
     We can thus instead output a lower-triangular matrix and multiplied by its transpose.
    - it allows to solve efficiently a system of linear equations :math:`Ax = b` without having to compute the inverse
     (and thus, the determinant). This is achieved in a 2-step way, by first computing :math:`Ly=b` for :math:`y` by
     forward substitution, and then computing :math:`L^\top x = y` by backward substitution.
    """

    def __init__(self, num_inputs, num_outputs, offset=1.e-4):
        """
        Initialize the full covariance generator.

        Args:
            num_inputs (int): size of the base output vector
            num_outputs (int): size of the action mean vector
            offset (float): small offset to be added to the diagonal elements of the covariance matrix such that
                it is positive definite.
        """
        super(FullCovarianceModule, self).__init__()
        # size of the mean vector
        self._dim = num_outputs

        # indices for lower triangular matrix
        self._idx = np.tril_indices(self._dim)

        # define diagonal offset such that the covariance is positive definite
        self._offset = offset * torch.diag(torch.ones(self._dim))

        # linear mapping between the base output vector / matrix to the covariance's lower triangular matrix
        model = torch.nn.Linear(in_features=num_inputs, out_features=len(self._idx[0]))
        self._model = init_orthogonal_weights_and_constant_biases(model)

    def forward(self, x):
        """Take as input the base output vector / matrix and return the full covariance matrix(ces).

        Args:
            x (torch.Tensor): base output vector / matrix of shape (N, B)

        Returns:
            torch.Tensor: covariance matrices (one covariance matrix for each sample) of shape (N, D, D)
        """
        # from base outputs to vector representing the elements of a triangular matrix
        x = self._model(x)  # shape (N,L) or (L,) where L=(D^2+D)/2

        # create lower triangular matrix
        if len(x.shape) > 1:
            covariance = torch.zeros(x.size(0), self._dim, self._dim)  # shape (N,D,D)

            # set the elements in the covariance matrix  # TODO: find a better way than a for-loop
            for i in range(len(covariance)):
                covariance[i][self._idx] = x[i]
                covariance[i] = torch.matmul(covariance[i], covariance[i].t()) + self._offset

            # add a small noise to the diagonal terms to be positive definite
            # covariance = covariance + self.threshold * torch.diag(torch.ones(self.dim))  # shape (N,D,D)
        else:
            covariance = torch.matmul(x, x.t()) + self._offset  # shape (D,D)

        # return the full covariance matrix
        return covariance


# define aliases for logits and probs module
FixedLogitsModule = FixedVectorModule
FixedProbsModule = FixedVectorModule
LogitsModule = VectorModule
ProbsModule = VectorModule


class DiscreteModule(torch.nn.Module):
    r"""Discrete probability module.

    Discrete probability module from which several discrete probability distributions (such as Bernoulli, Categorical,
    and others) inherit from.
    """

    __metaclass__ = ABCMeta

    def __init__(self, probs=None, logits=None):
        """
        Initialize the Discrete probability module.

        Args:
            probs (torch.nn.Module): event probabilities module.
            logits (torch.nn.Module): event logits module.
        """
        super(DiscreteModule, self).__init__()
        if probs is None and logits is None:
            raise ValueError("Expectingt the given 'probs' xor 'logits' to be different than None.")
        if probs is not None and logits is not None:
            raise ValueError("Expecting the given 'probs' xor 'logits' to be None.")

        if probs is not None:
            if not isinstance(probs, torch.nn.Module):
                raise TypeError("Expecting the probs to be an instance of `torch.nn.Module`, instead got: "
                                "{}".format(type(probs)))
            self._probs = probs
            self._logits = lambda x: None

        if logits is not None:
            if not isinstance(logits, torch.nn.Module):
                raise TypeError("Expecting the logits to be an instance of `torch.nn.Module`, instead got: "
                                "{}".format(type(logits)))
            self._logits = logits
            self._probs = lambda x: None

    @property
    def logits(self):
        """Return the logits module."""
        return self._logits

    @property
    def probs(self):
        """Return the probabilities module."""
        return self._probs


class CategoricalModule(DiscreteModule):
    r"""Categorical Module

    Type: discrete, multiple categories

    The Categorical module accepts as inputs the discrete logits or probabilities modules, and returns the categorical
    distribution (that inherits from `torch.distributions.Categorical`).

    Description: "A categorical distribution (also called a generalized Bernoulli distribution, multinoulli
    distribution) is a discrete probability distribution that describes the possible results of a random variable that
    can take on one of K possible categories, with the probability of each category separately specified." [1]

    Examples:
        >>> # fixed categorical
        >>> logits = FixedLogitsModule(torch.ones(5))
        >>> categorical = CategoricalModule(logits=logits)
        >>> probs = categorical(base_output)  # or categorical(output)
        >>> # flexible categorical
        >>> logits = LogitsModule(num_inputs=10, num_outputs=5)
        >>> categorical = CategoricalModule(logits=logits)
        >>> probs = categorical(base_output)
        >>> # identity categorical (just copy what will be given as inputs)
        >>> logits = IdentityModule()
        >>> categorical = CategoricalModule(logits=logits)
        >>> probs = categorical(output)

    References:
        [1] Categorical distribution: https://en.wikipedia.org/wiki/Categorical_distribution
    """

    def __init__(self, probs=None, logits=None):
        """
        Initialize the Categorical module.

        Args:
            probs (torch.nn.Module): event probabilities module.
            logits (torch.nn.Module): event logits module.
        """
        super(CategoricalModule, self).__init__(probs=probs, logits=logits)

    def forward(self, x):
        """Forward the given inputs :attr:`x`."""
        return CategoricalDistribution(probs=self.probs(x), logits=self.logits(x))


class BernoulliModule(DiscreteModule):
    r"""Bernoulli Module

    Type: discrete, binary

    "The Bernoulli distribution is the discrete probability distribution of a random variable which takes the value 1
    with probability :math:`p` and the value 0 with probability :math:`q = 1-p`, that is, the probability distribution
    of any single experiment that asks a yes/no question; the question results in a boolean-valued outcome, a single
    bit of information whose value is success with probability :math:`p` and failure with probability :math:`q`." [1]
    
    Examples:
        >>> # fixed categorical
        >>> logits = FixedLogitsModule(torch.ones(5))
        >>> bernoulli = BernoulliModule(logits=logits)
        >>> probs = bernoulli(base_output)  # or bernoulli(output)
        >>> # flexible categorical
        >>> logits = LogitsModule(num_inputs=10, num_outputs=5)
        >>> bernoulli = BernoulliModule(logits=logits)
        >>> probs = bernoulli(base_output)
        >>> # identity categorical (just copy what will be given as inputs)
        >>> logits = IdentityModule()
        >>> bernoulli = BernoulliModule(logits=logits)
        >>> probs = bernoulli(output)

    References:
        [1] Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
    """

    def __init__(self, probs=None, logits=None):
        """
        Initialize the Bernoulli module.

        Args:
            probs (torch.nn.Module): event probabilities module.
            logits (torch.nn.Module): event logits module.
        """
        super(BernoulliModule, self).__init__(probs=probs, logits=logits)
        self.logits = logits
        self.probs = probs

    def forward(self, x):
        """Forward the given inputs :attr:`x`."""
        return BernoulliDistribution(probs=self.probs(x), logits=self.logits(x))


class NormalModule(torch.nn.Module):
    r"""Normal Module

    Type: continuous

    The Normal module accepts as inputs the mean and standard deviation modules (i.e. that inherit from
    `torch.nn.Module`, and returns the Normal distribution (i.e. `torch.distributions.Normal`). For more information
    about this distribution, see the documentation of `torch.distributions.Normal`.

    Examples:
        >>> # fixed normal
        >>> mean = FixedMeanModule(mean=torch.zeros(5))
        >>> stddev = FixedStandardDeviationModule(stddev=torch.ones(5))
        >>> normal = NormalModule(mean=mean, stddev=stddev)
        >>> # most flexible normal
        >>> mean = MeanModule(num_inputs=10, num_outputs=5)
        >>> stddev = StandardDeviationModule(num_inputs=10, num_outputs=5)
        >>> normal = NormalModule(mean=mean, stddev=stddev)
        >>> # if the mean is already computed
        >>> mean = IdentityModule()
        >>> stddev = StandardDeviationModule(num_inputs=10, num_outputs=5)
        >>> normal = NormalModule(mean=mean, stddev=stddev)
    """

    def __init__(self, mean, stddev):
        """
        Initialize the Normal module.

        Args:
            mean (torch.nn.Module): mean module.
            stddev (torch.nn.Module): standard deviation module.
        """
        super(NormalModule, self).__init__()
        self.mean = mean
        self.stddev = stddev

    @property
    def mean(self):
        """Return the mean module."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        """Set the mean module."""
        if not isinstance(mean, torch.nn.Module):
            raise TypeError("Expecting the mean to be an instance of `torch.nn.Module`, instead got: "
                            "{}".format(type(mean)))
        self._mean = mean

    @property
    def stddev(self):
        """Return the stddev module."""
        return self._stddev

    @stddev.setter
    def stddev(self, stddev):
        """Set the stddev module."""
        if not isinstance(stddev, torch.nn.Module):
            raise TypeError("Expecting the stddev to be an instance of `torch.nn.Module`, instead got: "
                            "{}".format(type(stddev)))
        self._stddev = stddev

    def forward(self, *x):
        """Forward the given inputs :attr:`x`."""
        if len(x) == 1:
            x1, x2 = x[0], x[0]
        elif len(x) == 2:
            x1, x2 = x[0], x[1]
        else:
            raise ValueError("Expecting 1 or 2 inputs.")
        mean = self.mean(x1)
        stddev = self.stddev(x2)
        return torch.distributions.Normal(loc=mean, scale=stddev)


class GaussianModule(torch.nn.Module):
    r"""Gaussian Module

    Type: continuous

    The Gaussian module accepts as inputs the mean and covariance modules (i.e. that inherit from `torch.nn.Module`),
    and returns the multivariate Gaussian distribution (that inherits from `torch.distributions.MultivariateNormal`).
    For more information about this distribution, see the documentation of `pyrobolearn/distributions/gaussian.py`.

    Examples:
        >>> # fixed gaussian (that has a fixed mean and diagonal covariance)
        >>> mean = FixedMeanModule(mean=torch.zeros(5))
        >>> covariance = FixedDiagonalCovarianceModule(variances=torch.ones(5))
        >>> fixed_gaussian = GaussianModule(mean=mean, covariance=covariance)
        >>> probs = fixed_gaussian(base_output)  # or fixed_gaussian(output)
        >>> # most flexible gaussian (that learns the mean and the full covariance)
        >>> mean = MeanModule(num_inputs=10, num_outputs=5)
        >>> covariance = FullCovarianceModule(num_inputs=10, num_outputs=5)
        >>> full_gaussian = GaussianModule(mean=mean, covariance=covariance)
        >>> probs = full_gaussian(base_output)
        >>> # if the mean is already computed elsewhere
        >>> mean = IdentityModule()
        >>> covariance = FullCovarianceModule(num_inputs=5, num_outputs=5)
        >>> gaussian = GaussianModule(mean=mean, covariance=covariance)
        >>> probs = gaussian(output, base_output)  # inputs for the mean and covariance
    """

    def __init__(self, mean, covariance):
        """
        Initialize the Gaussian module.

        Args:
            mean (torch.nn.Module): mean module.
            covariance (torch.nn.Module): covariance module.
        """
        super(GaussianModule, self).__init__()
        self.mean = mean
        self.covariance = covariance

    @property
    def mean(self):
        """Return the mean module."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        """Set the mean module."""
        if not isinstance(mean, torch.nn.Module):
            raise TypeError("Expecting the mean to be an instance of `torch.nn.Module`, instead got: "
                            "{}".format(type(mean)))
        self._mean = mean

    @property
    def covariance(self):
        """Return the covariance module."""
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        """Set the covariance module."""
        if not isinstance(covariance, torch.nn.Module):
            raise TypeError("Expecting the covariance to be an instance of `torch.nn.Module`, instead got: "
                            "{}".format(type(covariance)))
        self._covariance = covariance

    def forward(self, *x):
        """Forward the given inputs :attr:`x`."""
        if len(x) == 1:
            x1, x2 = x[0], x[0]
        elif len(x) == 2:
            x1, x2 = x[0], x[1]
        else:
            raise ValueError("Expecting 1 or 2 inputs.")
        mean = self.mean(x1)
        covariance = self.covariance(x2)
        return GaussianDistribution(mean=mean, covariance=covariance)


class GaussianMixtureModule(torch.nn.Module):
    r"""Gaussian Mixture Module

    Type: continuous

    The Gaussian mixture module accepts as inputs the priors, means and covariances modules (i.e. that inherit from
    `torch.nn.Module`), and returns the Gaussian mixture distribution
    (that inherits from `torch.distributions.Distribution`).
    For more information about this distribution, see the documentation of `pyrobolearn/distributions/gmm.py`.

    Examples:
       >>> # fixed gmm (that has fixed priors, means and diagonal covariances)
       >>> mean1 = FixedMeanModule(mean=torch.zeros(5))
       >>> mean2 = FixedMeanModule(mean=torch.ones(5))
       >>> cov1 = FixedDiagonalCovarianceModule(variances=torch.ones(5))
       >>> cov2 = FixedDiagonalCovarianceModule(variances=0.5 * torch.ones(5))
       >>> priors = FixedProbsModule(torch.tensor([0.3, 0.7]))  # because 2 gaussians
       >>> fixed_gmm = GaussianMixtureModule(priors=priors, means=[mean1, mean2], covariances=[cov1, cov2])
       >>> probs = fixed_gmm(base_output)  # or fixed_gmm(output)
       >>> # most flexible gaussian (that learns the priors, means and full covariances)
       >>> mean1 = MeanModule(num_inputs=10, num_outputs=5)
       >>> mean2 = MeanModule(num_inputs=10, num_outputs=5)
       >>> cov1 = FullCovarianceModule(num_inputs=10, num_outputs=5)
       >>> cov2 = FullCovarianceModule(num_inputs=10, num_outputs=5)
       >>> priors = VectorModule(num_inputs=10, num_outputs=2)  # because 2 gaussians
       >>> full_gmm = GaussianMixtureModule(priors=priors, means=[mean1, mean2], covariances=[cov1, cov2])
       >>> probs = full_gmm(base_output)
       >>> # if one of the means is already computed elsewhere
       >>> mean1 = IdentityModule()
       >>> mean2 = MeanModule(num_inputs=10, num_outputs=5)
       >>> cov1 = FullCovarianceModule(num_inputs=5, num_outputs=5)
       >>> cov2 = FullCovarianceModule(num_inputs=5, num_outputs=5)
       >>> priors = VectorModule(num_inputs=10, num_outputs=2)  # because 2 gaussians
       >>> gmm = GaussianMixtureModule(priors=priors, means=[mean1, mean2], covariances=[cov1, cov2])
       >>> probs = gmm(output, base_output)  # inputs for the mean and covariance
    """

    def __init__(self, means, covariances, priors=None):
        """
        Initialize the Gaussian mixture module.

        Args:
            priors (torch.nn.Module): prior module.
            means ((list of) torch.nn.Module): mean modules.
            covariances ((list of) torch.nn.Module): covariance modules.
        """
        super(GaussianMixtureModule, self).__init__()
        self.priors = priors
        self.means = means
        self.covariances = covariances

    @property
    def priors(self):
        """Returns the prior module."""
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Set the priors."""
        if not isinstance(priors, torch.nn.Module):
            raise TypeError("Expecting the priors to be an instance of `torch.nn.Module`, instead got: "
                            "{}".format(type(priors)))
        self._priors = priors

    @property
    def means(self):
        """Return a list of mean modules."""
        return self._means

    @means.setter
    def means(self, means):
        """Set the mean modules."""
        if not isinstance(means, collections.Iterable):
            means = [means]
        for mean in means:
            if not isinstance(mean, torch.nn.Module):
                raise TypeError("Expecting the mean to be an instance of `torch.nn.Module`, instead got: "
                                "{}".format(type(mean)))
        self._means = means

    @property
    def covariances(self):
        """Return a list of covariance modules."""
        return self._covariances

    @covariances.setter
    def covariances(self, covariances):
        """Set the covariance modules."""
        if not isinstance(covariances, collections.Iterable):
            covariances = [covariances]
        for covariance in covariances:
            if not isinstance(covariance, torch.nn.Module):
                raise TypeError("Expecting the covariance to be an instance of `torch.nn.Module`, instead got: "
                                "{}".format(type(covariance)))
        self._covariances = covariances

    def forward(self, *x):
        """Forward the given inputs :attr:`x`."""
        if len(x) == 1:
            x1, x2 = x[0], x[0]
        elif len(x) == 2:
            x1, x2 = x[0], x[1]  # output and base_output
        else:
            raise ValueError("Expecting 1 or 2 inputs.")

        # TODO: the for-loop is not really efficient...
        means = [mean(x1) if isinstance(mean, IdentityModule) else mean(x2) for mean in self.means]
        covariances = [covariance(x2) for covariance in self.covariances]
        priors = self.priors(x2)
        return GaussianMixtureDistribution(priors=priors, means=means, covariances=covariances)
