#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the quaternion kernelized movement primitive class.

This file provides the Quaternion-KMP model [1].

References:
    - [1] "Generalized Orientation Learning in Robot Task Space", Huang et al., 2019
    - [2] "Kernelized Movement Primitives", Huang et al., 2017
    - [3] https://github.com/yanlongtu/robInfLib
"""

import numpy as np

from pyrobolearn.models.kmp import KMP

# import the quaternion transformation mapping for Quaternion-KMP
from pyrobolearn.utils.transformation import logarithm_map, exponential_map, get_quaternion_product


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Yanlong Huang (paper + Matlab)", "Brian Delhaisse (Python)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class QuaternionKMP(KMP):
    r"""Quaternion KMP

    The Quaternion KMP consists first to project the quaternion data :math:`q_i \in \mathbb{S}^3; \forall i`, where
    :math:`\mathbb{S}^3` is a unit sphere in :math:`\mathbb{R}^4`, into :math:`\mathbb{R}^3` using the logarithm map
    :math:`\log: \mathbb{S}^3 \rightarrow \mathbb{R}^3`. Then, the KMP is trained on the projected data and the
    predicted data is reprojected onto :math:`\mathbb{S}^3` using the exponential map :math:`\exp: \mathbb{R}^3
    \rightarrow \mathbb{S}^3`.

    Note that "the logarithmic map defined on :math:`\mathbb{S}^3` has no discontinuity boundary, just a singularity
    at a single quaternion :math:`\bm{q} = -1 + [0,0,0]^\top = (-1, [0,0,0]^\top)`." [2]

    Note also that the KMP is initialized here using a Gaussian mixture model [3].

    Warnings: The output of this model is expected to be the concatenation of orientations (expressed as quaternions
        [x,y,z,w]) and their angular velocities.

    References:
        - [1] "Generalized Orientation Learning in Robot Task Space", Huang et al., 2019
        - [2] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
        - [3] "Kernelized Movement Primitives", Huang et al., 2017
    """

    def __init__(self, kernel_fct=None):
        """
        Initialize the Quaternion KMP.

        Args:
            kernel_fct (None, callable): kernel function. If None, it will use the `RBF` kernel with a variance
                of 1, and a length scale of 2.
        """
        super(QuaternionKMP, self).__init__(kernel_fct=kernel_fct)

        # define the auxiliary quaternion
        self.auxiliary_quaternion = np.array([0., 0., 0., 1.])  # (x,y,z,w)

    def fit(self, X, Y, gmm=None, gmm_num_components=10, prior_reg=1.,  dist=None, database_threshold=1e-3,
            database_size_limit=100, sample_from_gmm=False, gmm_init='kmeans', gmm_reg=1e-8, gmm_num_iters=1000,
            gmm_convergence_threshold=1e-4, seed=None, verbose=True, block=True):
        r"""
        Fit the given data composed of inputs and outputs.

        This works by minimizing the KL-divergence between a parametric probabilistic discriminative model
        and the predicted output distribution of a reference probabilistic model. First, the reference model (e.g.
        a Gaussian mixture model) is trained on the given data (i.e. inputs :math:`x \in \mathbb{R}^{I} and outputs
        :math:`y \in \mathbb{R}^{O}`). A reference database is then constructed containing `N` data inputs with the
        corresponding output Gaussian distribution resulting from GMR given the data inputs.

        Then, a parametric model is given by :math:`y(x) = \Phi(x)^T w` where a Gaussian distribution is put on the
        weights :math:`w \in \mathbb{R}^{BO}` such that :math:`w \sim \mathcal{N}(\mu_w, \Sigma_w)`, and thus
        :math:`y(x) \sim \mathcal{N}(\Phi(x)^T \mu_w, \Phi(x)^T \Sigma_w \Phi(x))`. The matrix
        :math:`\Phi(x) \in \mathbb{R}^{BO \times O}` is a block diagonal matrix containing basis functions
        on its diagonal.

        The loss that is being minimized by KMP is given by:

        .. math::

            \mathcal{L}(\mu_w, \Sigma_w) = \sum_{n=1}^N KL[p(y|x_n;\theta) || p_{ref}(y | x_n)]
                + \tau ( (\mu_w^T\mu_w) + tr(\Sigma_w) )

        where :math:`\theta = \{\mu_w, \Sigma_w\}` are the parameters that are being optimized,
        :math:`p_{ref}(y | x_n) = \mathcal{N}(\mu_n, \Sigma_n)` is the predicted reference distribution
        (e.g. Gaussian by GMR), and :math:`\tau` is the prior regularization term.

        Once the parametric model has been optimized, the optimal mean and covariance of the weights are given by:

        .. math::

            \mu_w = \Omega (\Omega^T \Omega + \tau \Sigma)^{-1} \mu
            \Sigma_w = N (\Omega \Sigma \Omega^T + \tau I)^{-1}

        where :math:`\Omega = [\Phi(x_1) ... \Phi(x_N)] \in \mathbb{R}^{BO \times NO}`,
        :math:`\Sigma = blockdiag(\Sigma_1, ..., \Sigma_N) \in \mathbb{R}^{NO \times NO}`, and
        :math:`\mu = [\mu_1^T ... \mu_N^T]^T \in \mathbb{R}^{NO \times 1}`.

        Thus, the predicted output mean and covariance on a new input :math:`x^*` is given by:

        .. math::

            \mu_y &= \Phi(x^*)^T \mu_w = \Phi(x^*) \Omega (\Omega^T \Omega + \tau \Sigma)^{-1} \mu \\
            \Sigma_y &= \Phi(x^*)^T \Sigma_w \Phi(x^*) = N \Phi(x^*)^T (\Omega\Sigma\Omega^T+\tau I)^{-1} \Phi(x^*)

        And by using the kernel trick (and the Woodbury identity for the covariance), this resumes to:

        .. math::

            \mu_y &= k^* (K + \tau \Sigma)^{-1} \mu \\
            \Sigma_y &= \frac{N}{\tau} (k(x^*, x^*) - k^* (K + \tau \Sigma)^{-1} k^*^T)

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        Args:
            X (np.array[N,T,I], list of np.array[T,I]): input data matrix of shape NxTxI, where N is the number of
                trajectories, T is its length, and I is the input data dimension.
            Y (np.array[N,T,O], list of np.array[T,O]): corresponding output data matrix of shape NxTxO, where N is
                the number of trajectories, T is its length, and O is the output data dimension.
            gmm (None, GMM): the reference generative model. If None, it will create a GMM.
            gmm_num_components (int): the number of components for the underlying reference GMM.
            prior_reg (float): prior regularization term
            dist (callable, None): callable function which accepts two data points from X, and compute the distance
                between them. If None and `sample_from_gmm` is False, it will use the 2-norm.
            database_threshold (float): threshold associated with the `distance` argument above. If the distance between
                a new data point and data point in the database is below the threshold, it will be added to
                the database.
            database_size_limit (int): limit size of the database.
            sample_from_gmm (bool): If we should sample from the generative model to get the inputs to put in the
                database. If True, it doesn't use the `distance` and `database_threshold` parameters.
            gmm_init (str): how the Gaussians should be initialized. Possible values are 'random' or 'kmeans'.
            gmm_reg (float): regularization term for the GMM (that are added to the Gaussians)
            gmm_num_iters (int): the maximum number of iterations to train the reference model (GMM)
            gmm_convergence_threshold (float): convergence threshold when training the reference model (GMM)
            seed (int, None): random seed for the initialization and training of the GMM, and when sampling
            verbose (bool): if we should print details during the optimization process
            block (bool): if the size of the kernel matrix is bigger than 1000, it will ask for confirmation to
                continue. The kernel matrix has to be inversed, which has a time complexity of `O(N^3)` where
                `N` is the size of the kernel matrix.

        References:
            - [1] "Kernelized Movement Primitives", Huang et al., 2017
        """
        # map the output quaternions from S^3 to R^3
        Y = logarithm_map(Y[:, :, :4])

        # call the parent
        super(QuaternionKMP, self).fit(X=X, Y=Y, gmm=gmm, gmm_num_components=gmm_num_components, prior_reg=prior_reg,
                                       dist=dist, database_threshold=database_threshold,
                                       database_size_limit=database_size_limit, sample_from_gmm=sample_from_gmm,
                                       gmm_init=gmm_init, gmm_reg=gmm_reg, gmm_num_iters=gmm_num_iters,
                                       gmm_convergence_threshold=gmm_convergence_threshold, seed=seed,
                                       verbose=verbose, block=block)

    def predict(self, x):
        r"""
        Predict output mean :math:`\mu_y` given input data :math:`x^*`.

        .. math::

            \mu_y(x^*) &= k^* (K + \tau \Sigma)^{-1} \mu \\

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        Args:
            x (np.array[I], np.array[N,I]): new input data vector or matrix

        Returns:
            np.array[O], np.array[N,O]: output mean(s)
        """
        # predict the output means using the KMP in R^3
        means = super(QuaternionKMP, self).predict(x)

        # map the predicted output in R^3 to S^3 using the exponential map
        # The 3 first components of the predicted outputs represent the quaternion, while the last 3 represent the
        # angular velocities.
        if len(means.shape) == 1:
            quaternion = get_quaternion_product(exponential_map(means[:3]), self.auxiliary_quaternion)  # shape: (4,)
            means = np.concatenate(quaternion, means[3:])  # shape: (7,)
        else:
            quaternions = get_quaternion_product(exponential_map(means[:, :3]), self.auxiliary_quaternion)  # (N, 4)
            means = np.hstack((quaternions, means[:, 3:]))  # shape: (N,7)

        return means


# Tests
if __name__ == '__main__':
    pass
