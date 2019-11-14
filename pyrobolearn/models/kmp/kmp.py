#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the kernelized movement primitive class.

This file provides the Kernelized Movement Primitive (KMP) model, and uses the Gaussian mixture model as well as
the Gaussian distribution defined respectively in `gmm.py` and `gaussian.py`.

References:
    - [1] "Kernelized Movement Primitives", Huang et al., 2017
    - [2] https://github.com/yanlongtu/robInfLib
"""

import numpy as np
from scipy.linalg import block_diag
import copy

# from pyrobolearn.models.model import Model
from pyrobolearn.models.gmm import GMM, Gaussian

# to check Python version (if sys.version_info[0] < 3, then python 2)
import sys
if sys.version_info[0] < 3:  # Python 2
    input = raw_input  # redefine input to be raw_input


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Yanlong Huang (paper + Matlab)", "Brian Delhaisse (Python)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RBF(object):
    r"""
    RBF kernel.

    .. math:: k(x1, x2) = \sigma^2 * \exp(- ||x_1 - x_2||^2 / l)

    where :math:`x1` and :math:`x2` are two vectors, :math:`\sigma^2` is the variance, and :math:`l` is
    the length scale.
    """

    def __init__(self, variance=1., lengthscale=1.):
        """
        Initialize the RBF kernel.

        Args:
            variance (float): variance
            lengthscale (float): lengthscale
        """
        self.var = variance
        self.l = lengthscale

    def k(self, x1, x2=None):
        r"""
        Compute kernel function: :math:`k(x1, x2) = \sigma^2 * \exp(- ||x_1 - x_2||^2 / l)`

        where :math:`x1` and :math:`x2` are two vectors, :math:`\sigma^2` is the variance, and :math:`l` is
        the length scale.

        Args:
            x1 (float, np.array): 1st value
            x2 (float, np.array, None): 2nd value. if None, it will take x1.

        Returns:
            float: similarity measure between the two given values.
        """
        if x2 is None:
            x2 = x1
        diff = x1 - x2
        return self.var * np.exp(- np.inner(diff, diff) / self.l)

    def __call__(self, x1, x2=None):
        """Return output from kernel function"""
        return self.k(x1, x2)


class KMP(object):
    r"""Kernelized Movement Primitives

    Kernelized Movement Primitives allows to encode a movement/trajectory using kernels. The use of kernels makes it
    practical for high-dimensional inputs.

    KMP is a non-parametric (but a semi-parametric approach is often used to initialize it) probabilistic
    discriminative model. The performance of the KMP depends thus on the underlying probabilistic distribution from
    which it learns from (which is often a GMM, and thus it depends on the GMM initialization).

    KMP is well-suited for via-point, end-point, extrapolation and high-dimensional inputs problems. However, note that
    it does not scale well with high-dimensional outputs or long trajectories. The reason is because of the kernel
    matrix which has a shape of :math:`K \in \mathbb{R}^{TO \times TO}` where :math:`O` is the output dimension, and
    :math:`T` is the length of the reference trajectory (i.e. number of data points sampled from that trajectory).

    References:
        - [1] "Kernelized Movement Primitives", Huang et al., 2017
    """

    def __init__(self, kernel_fct=None, database=None):
        r"""
        Initialize the KMP.

        Args:
            kernel_fct (None, callable): kernel function. If None, it will use the `RBF` kernel with a variance
                of 1, and a length scale of 2.
            database (None, list): initial reference database. The database should be a list where each item is a
                tuple of 3 elements (the input state, the mean, and the covariance). That is, the reference database
                is given by: :math:`\[s_t, \hat{\mu}_t, \hat{\Sigma}_t \]_{t=1}^T` where :math:`T` is the length of
                a reference trajectory, :math:`s_t` is the input (vector/scalar) state, :math:`\hat{\mu}_t` is the
                reference mean, and :math:`\hat{\Sigma}_t` is the reference covariance matrix. By reference, we mean
                that after training your probabilistic discriminative model on multiple trajectories, you provide the
                predicted mean and covariance given the input state.
        """
        super(KMP, self).__init__()

        self._input_dim = 0   # input dimension
        self._output_dim = 0  # output dimension
        self.N = 0            # number of data points

        # set kernel fct
        self.K = kernel_fct if kernel_fct is not None else RBF(variance=1., lengthscale=2.)

        # reference database
        if database is None:
            self._database = []
        else:
            self._database = database

        # mean
        self.mu = None  # mean used for the mean prediction

        # Inverse Kernel matrix (useful when computing the prediction)
        self.lambda1 = 1.  # lambda in the paper for the mean prediction (in Huang's code, it is often set to 1)
        self.lambda2 = 60.  # lambda in the paper for the covariance prediction (in Huang's code, it is set to 60)
        self.K_inv1 = None  # inverse kernel matrix for the mean
        self.K_inv2 = None  # inverse kernel matrix for the covariance

        # translation vector and rotation matrix
        self.bias = 0
        self.rot = None

    ##############
    # Properties #
    ##############

    @property
    def kernel_fct(self):
        """Return the kernel fct used for KMP"""
        return self.K

    @property
    def database(self):
        """Return the reference database"""
        return self._database

    @property
    def input_dim(self):
        """Return the input dimension"""
        return self._input_dim

    @property
    def output_dim(self):
        """Return the output dimension"""
        return self._output_dim

    @property
    def bias_vector(self):
        """Return the bias vector added to the predicted mean by the KMP"""
        return self.bias

    @property
    def rotation_matrix(self):
        """Return the rotation matrix applied to the predicted mean and covariance by the KMP"""
        return self.rot

    @property
    def lambda1(self):
        """Return the prior regularization term for the mean."""
        return self._l1

    @lambda1.setter
    def lambda1(self, value):
        """Set the prior regularization term for the mean."""
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting the prior regularization term for the mean to be a scalar (int, float), but "
                            "got instead: {}".format(type(value)))
        if value <= 0.:
            raise ValueError("The prior regularization term needs to be strictly bigger than 0.")
        self._l1 = value

    # aliases
    prior_mean_regularization = lambda1
    mean_regularization = lambda1
    mean_reg = lambda1

    @property
    def lambda2(self):
        """Return the prior regularization term for the covariance."""
        return self._l2

    @lambda2.setter
    def lambda2(self, value):
        """Set the prior regularization term for the covariance."""
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting the prior regularization term for the covariance to be a scalar (int, float), "
                            "but got instead: {}".format(type(value)))
        if value <= 0.:
            raise ValueError("The prior regularization term needs to be strictly bigger than 0.")
        self._l2 = value

    # aliases
    prior_covariance_regularization = lambda2
    covariance_regularization = lambda2
    cov_reg = lambda2

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other):  # TODO: use deepcopy instead...
        """Copy the other KMP."""
        if not isinstance(other, KMP):
            raise TypeError("Expecting the other element to be an instance of `KMP`, but got instead: "
                            "{}".format(type(other)))
        kmp = KMP(kernel_fct=other.kernel_fct)
        kmp._database = copy.deepcopy(other.database)
        kmp.bias = other.bias
        kmp.rot = other.rot
        kmp.lambda1 = other.lambda1
        kmp.lambda2 = other.lambda2
        kmp.K_inv1 = np.copy(other.K_inv1)
        kmp.K_inv2 = np.copy(other.K_inv2)
        return kmp

    @staticmethod
    def is_parametric():
        """The KMP is a non-parametric model which uses a kernel."""
        return False

    @staticmethod
    def is_linear():
        """The KMP has no parameters, and thus has no linear parameters."""
        return False

    @staticmethod
    def is_recurrent():
        """The KMP is not a recurrent model where the output depends on the given input and previous outputs.
        Sequential data are encoded in the kernel."""
        return False

    @staticmethod
    def is_probabilistic():  # same as is_stochastic
        """The KMP returns a mean and covariance matrix which parametrizes a normal distribution"""
        return True

    @staticmethod
    def is_discriminative():
        r"""The KMP is a discriminative model which predicts :math:`p(y|x)` where :math:`x` is the input,
        and :math:`y` is the output"""
        return True

    @staticmethod
    def is_generative():
        """The KMP is not a generative model, and thus we can not sample from it"""
        return False

    @staticmethod
    def create_reference_database(X, Y, gmm=None, gmm_num_components=10, distance=None, database_threshold=1e-3,
                                  database_size_limit=100, sample_from_gmm=False, gmm_init='kmeans', gmm_reg=1e-8,
                                  gmm_num_iters=1000, gmm_convergence_threshold=1e-4, seed=None, verbose=False,
                                  block=True):
        r"""
        Create reference database from the data. This database contains a list of input data with their
        corresponding predicted output distribution by the reference model (which in this case is a GMM).

        That is from several trajectories :math:`\{ \{ s_{t,n}, \xi_{t,n} \}_{t=1}^T_n \}_{n=1}^N`, it computes the
        reference database :math:`\{ s_t, \hat{\mu}_t, \hat{\Sigma}_t \}_{t=1}^T`, where :math:`N` is the number of
        trajectories, :math:`T_n` is the length of the trajectory :math:`n`, :math:`s` is the input, :math:`\xi` is
        the output, :math:`\hat{\mu}_t` is the reference mean, and :math:`\hat{\Sigma}_t` is the reference covariance
        matrix. By reference, we mean that after training the probabilistic discriminative model on multiple
        trajectories, the predicted mean and covariance given each input become the references.

        Args:
            X (np.array[N,T,I], list[np.array[T,I]]): input data matrix of shape NxTxI, where N is the number of
                trajectories, T is its length, and I is the input data dimension.
            Y (np.array[N,T,O], list[np.array[T,O]]): corresponding output data matrix of shape NxTxO, where N is
                the number of trajectories, T is its length, and O is the output data dimension.
            gmm (None, GMM): the reference generative model. If None, it will create a GMM.
            gmm_num_components (int): the number of components for the underlying reference GMM.
            distance (callable, None): callable function which accepts two data points from X, and compute the distance
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
                continue. The kernel matrix has to be inverted, which has a time complexity of `O(N^3)` where
                `N` is the size of the kernel matrix.

        Returns:
            list[(np.ndarray, Gaussian)]: database which is a list of tuples where each one contains an input data
                array and the corresponding predicted output Gaussian (by GMR)
        """
        # TODO: replace gmm by joint generative model

        # check given arguments
        X, Y = np.asarray(X), np.asarray(Y)
        if X.shape[:2] != Y.shape[:2]:
            if X.shape[0] != Y.shape[0]:
                raise ValueError("The number of trajectories are different between the input and output data")
            else:
                raise ValueError("The length of trajectories between the input and output data do not match.")

        # get useful variables (number of trajectories, their length, input dimension, output dimension)
        N, T, I = X.shape
        O = Y.shape[2]
        N_tot = N * T  # total number of data points

        # create GMM
        if gmm is None:
            gmm = GMM(num_components=gmm_num_components)

        # reshape the data to be N_tot x D where D = I + O
        data = np.dstack((X, Y))  # shape: NxTxD
        data = data.reshape(-1, I + O)  # shape: NTxD

        if verbose:
            print("Training the GMM...")

        # train gmm
        gmm.fit(data, reg=gmm_reg, num_iters=gmm_num_iters, threshold=gmm_convergence_threshold, init=gmm_init,
                seed=seed, verbose=verbose)

        if verbose:
            print("GMM trained")

        # create reference database
        database = []

        # if sample from GMM
        if sample_from_gmm:
            database = gmm.sample(size=database_size_limit)[:, range(I)]  # shape: NdxI

        # use the distance function to check if we should add the input data into the database
        else:
            # define distance function
            if distance is None:
                def distance(x1, x2):
                    return np.linalg.norm(x1 - x2)

            # check inputs to put in the reference database (time complexity: O((NT)^2))
            for x_traj in X:
                for x_curr in x_traj:
                    # compare current input with previous inputs, and add in database if unique enough
                    can_add = True
                    for x_prev in database:
                        if distance(x_curr, x_prev) < database_threshold:
                            can_add = False
                            break
                    if can_add:
                        database.append(x_curr)

            # if the size of the database is bigger than database size limit, sample uniformly from it
            if len(database) > database_size_limit:
                idx = np.random.choice(list(range(len(database))), size=database_size_limit, replace=False)
                database = database[idx]

        if verbose:
            print("Creating database...")

        # update database to also contain prediction from GMR
        database = [(x, gmm.condition(x, idx_out=list(range(I, I+O)),
                                      idx_in=list(range(I))).approximate_by_single_gaussian())
                    for x in database]

        if verbose:
            print("Database created...")

        # return constructed reference database
        return database

    @staticmethod
    def get_reference_database(x, means=None, covariances=None, gaussians=None):
        """
        Get reference database from the state inputs, means and covariances (gaussians).

        Args:
            x (np.array[float[T,I]]): input data matrix of shape TxI, where T is the length of a trajectory, and I is
              the input data dimension.
            means (np.array[float[T,O]]): list of means.
            covariances (np.array[float[T,O,O]]): list of covariances.
            gaussians (list[Gaussian]): list of Gaussian.

        Returns:
            list[(np.ndarray, Gaussian)]: database which is a list of tuples where each one contains an input data
                array and the corresponding predicted output Gaussian (by GMR)
        """
        if gaussians is None:
            if means is None:
                raise ValueError("If the gaussians are not provided, the means are required.")
            if covariances is None:
                raise ValueError("If the gaussians are not provided, the covariances are required.")
            gaussians = [Gaussian(mean=mean, covariance=covariance) for mean, covariance in zip(means, covariances)]
        database = [(xi, gaussian) for xi, gaussian in zip(x, gaussians)]
        return database

    def set_reference_database(self, x, means=None, covariances=None, gaussians=None):
        """
        Set reference database from the state inputs, means and covariances (gaussians).

        Args:
            x (np.array[float[T,I]]): input data matrix of shape TxI, where T is the length of a trajectory, and I is
              the input data dimension.
            means (np.array[float[T,O]]): list of means.
            covariances (np.array[float[T,O,O]]): list of covariances.
            gaussians (list[Gaussian]): list of Gaussian.

        Returns:
            list[(np.ndarray, Gaussian)]: database which is a list of tuples where each one contains an input data
                array and the corresponding predicted output Gaussian (by GMR)
        """
        self._database = self.get_reference_database(x, means=means, covariances=covariances, gaussians=gaussians)

    @staticmethod
    def combine(x, kmps, frames):
        r"""
        Combine different local KMPs.

        Warnings: This assumes that the input and output data can be mapped from local frames to a base frame
        by an affine transformation. This does not work with inputs or outputs that represents something else
        than coordinates. For instance, it does not work if the inputs are images or sensor values.

        Args:
            x (np.array[float[I]], np.array[float[N,I]]): new input data vector or matrix
            kmps (KMP, list of KMP): list of local KMPs
            frames (tuple, list of tuples): list of tuples where each tuple contains a rotation matrix and a bias
                translation vector

        Returns:
            Gaussian: resulting predicted Gaussian
        """
        if len(kmps) != len(frames):
            raise ValueError("The number of local frames is different from the number of KMPs")

        gaussians = []
        for kmp, frame in zip(kmps, frames):
            # get rotation matrix and translation vector
            A, b = frame
            if not isinstance(A, np.ndarray) or len(A.shape) != 2:
                raise TypeError("Expecting A to be a rotation matrix (2D array)")
            if not isinstance(b, np.ndarray) or len(A.shape) != 1:
                raise TypeError("Expecting b to be a translation vector (1D array)")

            # predict the gaussian distribution in the local frame
            gaussian = kmp.predict_proba(x, return_gaussian=True)

            # project back the distribution on the base frame
            gaussian = A * gaussian + b

            gaussians.append(gaussian)

        # compute the product of all gaussians which is the optimal solution
        gaussian = np.prod(gaussians)
        return gaussian

    ###########
    # Methods #
    ###########

    def fit(self, X, Y, gmm=None, gmm_num_components=10, mean_reg=1., covariance_reg=1., distance=None,
            database_threshold=1e-3, database_size_limit=100, sample_from_gmm=False, gmm_init='kmeans', gmm_reg=1e-8,
            gmm_num_iters=1000, gmm_convergence_threshold=1e-4, seed=None, verbose=False, block=True):
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
                + \lambda ( (\mu_w^T\mu_w) + tr(\Sigma_w) )

        where :math:`\theta = \{\mu_w, \Sigma_w\}` are the parameters that are being optimized,
        :math:`p_{ref}(y | x_n) = \mathcal{N}(\mu_n, \Sigma_n)` is the predicted reference distribution
        (e.g. Gaussian by GMR), and :math:`\lambda` is the prior regularization term.

        Once the parametric model has been optimized, the optimal mean and covariance of the weights are given by:

        .. math::

            \mu_w = \Omega (\Omega^T \Omega + \lambda \Sigma)^{-1} \mu
            \Sigma_w = N (\Omega \Sigma \Omega^T + \lambda I)^{-1}

        where :math:`\Omega = [\Phi(x_1) ... \Phi(x_N)] \in \mathbb{R}^{BO \times NO}`,
        :math:`\Sigma = blockdiag(\Sigma_1, ..., \Sigma_N) \in \mathbb{R}^{NO \times NO}`, and
        :math:`\mu = [\mu_1^T ... \mu_N^T]^T \in \mathbb{R}^{NO \times 1}`.

        Thus, the predicted output mean and covariance on a new input :math:`x^*` is given by:

        .. math::

            \mu_y &= \Phi(x^*)^T \mu_w = \Phi(x^*) \Omega (\Omega^T \Omega + \lambda \Sigma)^{-1} \mu \\
            \Sigma_y &= \Phi(x^*)^T \Sigma_w \Phi(x^*) = N \Phi(x^*)^T (\Omega\Sigma\Omega^T+\lambda I)^{-1} \Phi(x^*)

        And by using the kernel trick (and the Woodbury identity for the covariance), this resumes to:

        .. math::

            \mu_y &= k^* (K + \lambda \Sigma)^{-1} \mu \\
            \Sigma_y &= \frac{N}{\lambda} (k(x^*, x^*) - k^* (K + \lambda \Sigma)^{-1} k^*^T)

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        Args:
            X (np.array[N,T,I], list[np.array[T,I]]): input data matrix of shape NxTxI, where N is the number of
                trajectories, T is its length, and I is the input data dimension.
            Y (np.array[N,T,O], list[np.array[T,O]]): corresponding output data matrix of shape NxTxO, where N is
                the number of trajectories, T is its length, and O is the output data dimension.
            gmm (None, GMM): the reference generative model. If None, it will create a GMM.
            gmm_num_components (int): the number of components for the underlying reference GMM.
            mean_reg (float): prior regularization term for the mean that is multiplied by the covariance in the KMP
                (see lambda symbol in the paper [1]).
            covariance_reg (float): prior regularization term for the covariance that is multiplied by the covariance
                in the KMP (see lambda symbol in the paper [2]).
            distance (callable, None): callable function which accepts two data points from X, and compute the distance
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
        # TODO: replace gmm by joint generative model

        # create reference database
        self._database = self.create_reference_database(X, Y, gmm=gmm, gmm_num_components=gmm_num_components,
                                                        distance=distance, database_threshold=database_threshold,
                                                        database_size_limit=database_size_limit,
                                                        sample_from_gmm=sample_from_gmm, gmm_init=gmm_init,
                                                        gmm_reg=gmm_reg, gmm_num_iters=gmm_num_iters,
                                                        gmm_convergence_threshold=gmm_convergence_threshold,
                                                        seed=seed, verbose=verbose, block=block)

        # compute kernel inverse from database
        K, K_inv1, K_inv2 = self.learn_from_database(self._database, mean_reg=mean_reg, covariance_reg=covariance_reg,
                                                     verbose=verbose, block=block)

        self.K_inv1 = K_inv1     # shape: NOxNO
        self.K_inv2 = K_inv2     # shape: NOxNO

    # aliases
    learn = fit
    imitate = fit

    def learn_from_database(self, database=None, mean_reg=1., covariance_reg=1., verbose=False, block=True):
        r"""
        Learn the Kernel matrix from the database. Specifically, it computes :math:`K` and
        :math:`(K + \lambda \Sigma)^{-1}`. The latter is because this is used for the prediction part; for the
        predicted mean and covariance, and is better to compute it during the learning phase than the prediction phase.

        Args:
            database (list of tuples): list of tuples which contain the input data array and the associated predicted
                output distribution by the reference model.
            mean_reg (float): prior regularization term for the mean that is multiplied by the covariance in the KMP
                (see lambda symbol in the paper [1]).
            covariance_reg (float): prior regularization term for the covariance that is multiplied by the covariance
                in the KMP (see lambda symbol in the paper [2]).
            verbose (bool): if we should print details during the optimization process
            block (bool): if the size of the kernel matrix is bigger than 1000, it will ask for confirmation to
                continue. The kernel matrix has to be inversed, which has a time complexity of `O(N^3)` where
                `N` is the size of the kernel matrix.

        Returns:
            np.array[NO,NO]: Kernel matrix :math:`K`
            np.array[NO,NO]: Inverse Kernel matrix :math:`(K + \lambda_1 \Sigma)^-1` for mean prediction.
            np.array[NO,NO]: Inverse Kernel matrix :math:`(K + \lambda_2 \Sigma)^-1` for covariance prediction.
        """
        # Quick checks
        if database is None:
            database = self.database
        if len(database) == 0:
            raise ValueError("There are no elements in the database")

        # output dimension and size of database
        output_dim = database[0][1].size
        N = len(self._database)

        # check the size of the kernel matrix
        if N * output_dim > 1000 and verbose:
            print("Warning: trying to inverse a {} by {} 2D matrix... This could be computationally "
                  "expensive...".format(N * output_dim, N * output_dim))
            if block:
                input("Please press enter to continue with the inversion of the matrix. Ctrl+C to stop "
                      "the program")

        # remember variables for prediction
        self.N, self.lambda1, self.lambda2 = len(database), mean_reg, covariance_reg
        self._output_dim = output_dim
        self._input_dim = database[0][0].size

        # compute mean, covariance, and kernel from database
        self.mu = np.array([gaussian.mean for _, gaussian in self.database]).reshape(-1)  # shape: NO x 1
        cov = block_diag(*[gaussian.cov for _, gaussian in self.database])  # shape: NOxNO
        I_O = np.identity(output_dim)  # shape: OxO
        K = np.vstack([np.hstack([self.K(xi, xj) * I_O for xj, _ in self.database])
                      for xi, _ in self.database])  # shape: NOxNO

        # compute kernel inverse
        if verbose:
            print("Mean shape: {}".format(self.mu.shape))
            print("Covariance shape: {}".format(cov.shape))
            print("I_O shape: {}".format(I_O.shape))
            print("Inversing the kernel matrices with shape: {}".format(K.shape))

        K_inv1 = np.linalg.inv(K + mean_reg * cov)  # shape: NOxNO
        K_inv2 = K_inv1 if mean_reg == covariance_reg else np.linalg.inv(K + covariance_reg * cov)  # shape: NOxNO

        if verbose:
            print("The kernel matrices have been inversed...")

        # return kernel and kernel inverse
        return K, K_inv1, K_inv2

    def loss(self):
        r"""
        Compute the KL loss between the fitted KMP and the reference database.

        .. math::

            \mathcal{L} = \sum_{n=1}^N KL[\mathcal{N}(\mu_n^*, \Sigma_n^*) || \mathcal{N}_{ref}(\mu_n, \Sigma_n)]

        where :math:`\mu_n^* = k^* (K + \lambda \Sigma)^{-1} \mu` and
        :math:`\Sigma_n^* = \frac{N}{\lambda} (k(x^*, x^*) - k^* (K + \lambda \Sigma)^{-1} k^*^T)` are the predicted
        mean and covariance by the KMP, and :math:`\mu_n` and :math:`\Sigma_n` are the predicted mean and covariance
        by GMR.

        Returns:
            float: KL loss
        """
        loss = 0.

        # go through the database
        for x, gaussian in self.database:
            # predict gaussian by KMP
            kmp_gaussian = self.predict_proba(x, return_gaussian=True)

            # compute KL divergence between gaussian from database (which is a result of GMR), and the gaussian
            # predicted by the KMP
            kl_loss = gaussian.kl_divergence(kmp_gaussian)

            # add individual loss
            loss += kl_loss

        # return total loss
        return loss

    def _compute_k(self, x):
        r"""
        Compute the k matrix between the given new input and the inputs from the reference database.

        Args:
            x (np.array[I], np.array[N,I]): input data vector or matrix

        Returns:
            np.array: k vector
        """
        # compute k vector (which compares given input data with previous ones)
        I = np.identity(self.output_dim)
        # k = np.array([self.K(x, x_prev) * I for x_prev, _ in self.database])    # shape: NxOxO
        # k = k.reshape(-1, 1).T     # shape: OxNO
        k = np.hstack([self.K(x, x_prev) * I for x_prev, _ in self.database])  # shape: OxNO
        return k

    def predict(self, x):
        r"""
        Predict output mean :math:`\mu_y` given input data :math:`x^*`.

        .. math::

            \mu_y(x^*) &= k^* (K + \lambda \Sigma)^{-1} \mu \\

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        Args:
            x (np.array[I], np.array[N,I]): new input data vector or matrix

        Returns:
            np.array[O], np.array[N,O]: output mean(s)
        """
        # if only one sample
        if len(x.shape) == 1:
            x = [x]

        # compute predicted mean(s)
        means = []
        for xi in x:
            # compute k vector (which compares given input data with previous ones)
            k = self._compute_k(xi)

            # return mean
            mean = k.dot(self.K_inv1).dot(self.mu)
            means.append(mean)

        # return the same shape as input
        means = np.array(means)
        if means.shape[0] == 1:
            means = means[0]

        # return the predicted mean(s)
        return means

    def predict_proba(self, x, return_gaussian=True):
        r"""
        Predict the probability of output :math:`\mathcal{N}(\mu_y, \Sigma_y)` given input data :math:`x^*`.

        .. math::

            \mu_y(x^*) &= k^* (K + \lambda \Sigma)^{-1} \mu \\
            \Sigma_y(x^*) &= \frac{N}{\lambda} (k(x^*, x^*) - k^* (K + \lambda \Sigma)^{-1} k^*^T)

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        Args:
            x (np.array[I], np.array[N,I]): input data vector or matrix
            return_gaussian (bool): if True, it will return a list of Gaussians. Otherwise, it will return the means
              and covariances.

        Returns:
            if return_gaussian:
                Gaussian, or list of Gaussian: gaussian(s) for each input
            else:
                np.array[O], np.array[N,O]: output mean(s)
                np.array[O,O], np.array[N,O]: output covariance(s)
        """
        # if only one sample
        only_one_sample = False
        if isinstance(x, (int, float)):
            x = np.array([x])
        if len(x.shape) == 1:
            only_one_sample = True
            x = [x]

        # useful variables
        coeff = self.N / self.cov_reg
        I = np.identity(self.output_dim)

        # compute predicted mean(s) and covariance(s)
        means, covs = [], []
        for xi in x:
            # compute k vector
            k = self._compute_k(xi)
            k_input = self.K(xi, xi) * I

            # compute mean and covariance
            mean = k.dot(self.K_inv1).dot(self.mu)
            cov = coeff * (k_input - k.dot(self.K_inv2).dot(k.T))

            means.append(mean)
            covs.append(cov)

        # return the same shape as input
        means, covs = np.array(means), np.array(covs)
        if only_one_sample:
            means, covs = means[0], covs[0]

        # if need to return Gaussian(s)
        if return_gaussian:
            if only_one_sample:
                return Gaussian(mean=means, covariance=covs)
            return [Gaussian(mean=mean, covariance=cov) for mean, cov in zip(means, covs)]

        # else, return mean(s) and covariance(s)
        return means, covs

    def modulate(self, x, y_mean, y_cov, distance=None, threshold=1, update_database=False, mean_reg=1.,
                 covariance_reg=1., verbose=True, block=True):
        r"""
        Modulate the prediction given new data point with their associated covariances.

        Warnings: once the KMP has been modulated, the user has to relearn it on the original database if he/she
        wishes to predict again on the old data. This can be done by calling `kmp.learn_from_database(kmp.database)`,
        where `kmp.database` is the original database.

        Args:
            x (np.array[I], np.array[N,I]): input data vector or matrix
            y_mean (np.array[O], np.array[N,O]): mean of new data point(s)
            y_cov (np.array[O,O], np.array[N,O,O]): covariance of new data point(s). A small covariance means the user
                wants a high precision around the new data point.
            distance (callable, None): callable function which accepts two data points from X, and compute the distance
                between them. If None and `sample_from_gmm` is False, it will use the 2-norm.
            threshold (float): threshold associated with the `distance` argument above. If the distance between
                a new data point and data point in the database is below the threshold, it will be added to
                the database.
            update_database (bool): If True, it will modify permanently the original database by including the new
                given points.
            mean_reg (float): prior regularization term for the mean that is multiplied by the covariance in the KMP
                (see lambda symbol in the paper [1]).
            covariance_reg (float): prior regularization term for the covariance that is multiplied by the covariance
                in the KMP (see lambda symbol in the paper [2]).
            verbose (bool): if we should print details during the optimization process
            block (bool): if the size of the kernel matrix is bigger than 1000, it will ask for confirmation to
                continue. The kernel matrix has to be inversed, which has a time complexity of `O(N^3)` where
                `N` is the size of the kernel matrix.
        """
        # quick checks
        if len(self.database) == 0:
            raise ValueError("There are no elements in the reference database")
        if len(x.shape) == 1:
            x = [x]
        if len(y_mean.shape) == 1:
            y_mean = [y_mean]
        if len(y_cov.shape) == 2:
            y_cov = [y_cov]
        if len(x) != len(y_mean):
            raise ValueError("The number of input data points does not match the number of output data points")
        if len(y_mean) != len(y_cov):
            raise ValueError("The number of means and covariances for the output data points doesn't match")

        # define distance function
        if distance is None:
            def distance(x1, x2):
                return np.linalg.norm(x1 - x2)

        # copy database
        database = copy.deepcopy(self.database)

        # for each new input
        for xi, yi, y_ci in zip(x, y_mean, y_cov):
            # check the closest input inside the reference database  (time complexity: O((NT)^2))
            idx_closest = 0
            x_closest = database[idx_closest][0]
            dist_closest = distance(x_closest, xi)
            for idx, (x_curr, _) in enumerate(database):
                # if the current distance between the new point and the current point is smaller than the previous
                # closest one, update the closest point
                dist_curr = distance(x_curr, xi)
                if dist_curr < dist_closest:
                    idx_closest = idx
                    x_closest = x_curr
                    dist_closest = distance(x_closest, xi)

            # check with the threshold if the closest point should be replaced by the new input data point,
            # or if the new point should just be appended in the database
            gaussian = Gaussian(mean=yi, covariance=y_ci)
            if dist_closest < threshold:
                database[idx_closest] = (xi, gaussian)
            else:
                database.append((xi, gaussian))

        # compute kernel inverse from the extended database
        K, K_inv1, K_inv2 = self.learn_from_database(database, mean_reg=mean_reg, covariance_reg=covariance_reg,
                                                     verbose=verbose, block=block)
        self.K_inv1 = K_inv1  # shape: NOxNO
        self.K_inv2 = K_inv2  # shape: NOxNO

        if update_database:
            self._database = database

        # return the extended database
        return database

    # alias
    add_via_points = modulate

    def superpose(self, databases, priorities, update_database=False, mean_reg=1., covariance_reg=1.,
                  verbose=True, block=True):
        r"""
        Superpose different trajectories based on priorities.

        This is given by the following optimization:

        .. math::

            \mathcal{L} = \sum_{n=1}^N \sum_{l=1}^L \gamma_{n,l} KL[p(y|x_n;\theta) || p^l_{ref}(y | x_n)]

        where :math:`L` is the total number of trajectories (i.e. databases), :math:`\gamma_{n,l} \in ]0,1[` is
        the associated priority with each sample and trajectory and respects :math:`\sum_{l=1}^L \gamma_{n,l} = 1`.

        The optimal solution is given by the product of :math:`L` Gaussians whose mean and covariance are
        predicted by their corresponding KMP.

        Args:
            databases (list[list[(np.ndarray, Gaussian)]]): list of database where each database is a list of tuples
                where each one contains an input data array and the corresponding predicted output Gaussian (by GMR).
                The databases have the same size, and the same input arrays in the same order.
            priorities(np.array[L,N]]): list of priorities (float) for each point in each database.
            update_database (bool): If True, it will modify permanently the original database by including the new
                given points.
            mean_reg (float): prior regularization term for the mean that is multiplied by the covariance in the KMP
                (see lambda symbol in the paper [1]).
            covariance_reg (float): prior regularization term for the covariance that is multiplied by the covariance
                in the KMP (see lambda symbol in the paper [2]).
            verbose (bool): if we should print details during the optimization process
            block (bool): if the size of the kernel matrix is bigger than 1000, it will ask for confirmation to
                continue. The kernel matrix has to be inversed, which has a time complexity of `O(N^3)` where
                `N` is the size of the kernel matrix.
        """
        # quick checks
        if not isinstance(databases, (tuple, list, np.ndarray)):
            raise TypeError("Expecting a list of databases (i.e. list[list[(np.ndarray, Gaussian)]])")
        if len(databases) <= 0:
            raise ValueError("Expecting a non empty list of databases")
        if len(databases) != len(priorities):
            raise ValueError("The number of databases does not match with the number of trajectories")
        L, N = len(databases), len(databases[0])
        databases = np.array(databases)     # shape: LxNx2
        priorities = np.array(priorities)   # shape: LxN
        if priorities.shape != (L, N):
            raise ValueError("Expecting the priorities to be of shape (L,N) where L is the number of databases, "
                             "and N is the number of elements in these databases.")
        if not np.allclose(np.sum(priorities, axis=0), np.ones(L)):
            raise ValueError("The priorities should sum to one: np.sum(priorities, axis=0) == np.ones(L)")
        # TODO: check if same input

        # create mixed reference database
        mixed_database = []
        for i in range(N):
            priority = priorities[:, i]    # shape: L
            database = databases[:, i, 1]     # shape: L
            x_input = databases[0, i, 0]

            # quick check if similar input for each database
            for j in range(1, L):
                if np.allclose(databases[j-1, i, 0], databases[j, i, 0]):
                    raise ValueError("The element {} in the database {} and {} are different input "
                                     "arrays".format(i, j-1, j))

            # create rescaled gaussians
            gaussians = [Gaussian(mean=g.mean, covariance=g.cov / priority)
                         for priority, g in zip(priorities, database)]

            # take the product
            gaussians = np.prod(gaussians)

            # add the result in the database
            mixed_database.append((x_input, gaussians))

        # compute kernel inverse from the extended database
        K, K_inv1, K_inv2 = self.learn_from_database(mixed_database, mean_reg=mean_reg, covariance_reg=covariance_reg,
                                                     verbose=verbose, block=block)
        self.K_inv1 = K_inv1  # shape: NOxNO
        self.K_inv2 = K_inv2  # shape: NOxNO

        if update_database:
            self._database = mixed_database

        # return the mixed reference database
        return mixed_database

    # def create_local_databases(self, frames, global_database=None):
    #     """
    #     Create local databases from the global one, and return them. If the user wishes to learn local KMPs,
    #     he/she can create several KMP and then for each one of them, call the method `learn_from_database()` while
    #     providing the local database to it.
    #
    #     Args:
    #         frames:
    #         global_database:
    #
    #     Returns:
    #
    #     """
    #     pass

    def multiply(self, rotation_matrix):
        r"""
        Multiply the prediction of a KMP by a square matrix :math:`A`. The prediction of the KMP given a new input
        :math:`x` will now be given by: :math:`\mathcal{N}(A \mu(x), A \Sigma(x) A^T)`, where :math:`\mu` and
        :math:`\Sigma` are the original mean and covariance predicted by KMP.

        Args:
            rotation_matrix (np.array[O,O]): 2D rotation matrix of shape OxO, where O is the dimension of the output.

        Returns:
            KMP: resulting KMP
        """
        R = rotation_matrix
        if isinstance(R, np.ndarray):
            if len(R.shape) != 2:
                raise ValueError("Expecting the numpy array to be a 2D array, instead got shape:"
                                 " {}".format(R.shape))
            if R.shape != (self.output_dim, self.output_dim):
                raise ValueError("Size mismatch: the dimension of the predicted output by the KMP is {}, so expecting "
                                 " a 2D array of shape {} but got instead {}"
                                 "".format(self.output_dim, (self.output_dim, self.output_dim), R.shape))

            # copy KMP and rotate it
            kmp = KMP(kernel_fct=self.kernel_fct, database=self.database)
            kmp.rot = R.dot(self.rot)

            # return kmp
            return kmp
        raise TypeError("Expecting a rotation matrix")

    def add(self, bias_vector):
        r"""
        Add a vector :math:`b` to the prediction of a KMP. The prediction of the KMP given a new input :math:`x` will
        now be given by: :math:`\mathcal{N}(\mu(x) + b, \Sigma(x))`, where :math:`\mu` and :math:`\Sigma` are
        the original mean and covariance predicted by KMP.

        Args:
            bias_vector (np.array[O], float, int): 1D bias vector of the size of the output dimension. If an integer
                or a float number is given, it will create a vector of the size of the output dimension with the given
                value.

        Returns:
            KMP: resulting KMP
        """
        bias = bias_vector
        if isinstance(bias, (float, int)):
            bias = np.array([bias]*self.output_dim, dtype=np.float)

        if isinstance(bias, np.ndarray):
            if len(bias.shape) != 1:
                raise ValueError("Expecting the numpy array to be a 1D array, instead got shape:"
                                 " {}".format(bias.shape))
            if bias.shape[0] != self.output_dim:
                raise ValueError("Size mismatch: the dimension of the predicted output by the KMP is {} but the "
                                 "dimension of the given vector is {}".format(self.output_dim, bias.shape[0]))

            # copy KMP and add bias vector
            kmp = KMP(kernel_fct=self.kernel_fct, database=self.database)
            kmp.bias = self.bias + bias

            # return kmp
            return kmp
        raise TypeError("Expecting the other element to be a vector")

    def affine_transform(self, A, b=None):
        r"""
        Perform an affine transformation on the predicted output by the KMP. That is, given a new input :math:`x^*`,
        instead of predicting:

        .. math::

            \mu_y(x^*) &= k^* (K + \lambda \Sigma)^{-1} \mu \\
            \Sigma_y(x^*) &= \frac{T}{\lambda} (k(x^*, x^*) - k^* (K + \lambda \Sigma)^{-1} k^*^T)

        where :math:`K(X,X) \in \mathbb{R}^{NO \times NO}` is the kernel matrix,
        :math:`k^* = [k(x^*, x_1) ... k(x^*,x_N)] \in \mathbb{R}^{O \times NO}` is the kernel evaluated
        on the new input, and where :math:`k(x_i, x_j) = \hat{k}(x_i, x_j) I_O` with the identity matrix
        `I_O \in \mathbb{O \times O}` and :math:`\hat{k}(x_i, x_j)` the kernel function.

        it will predict:

        .. math::

            \mu(x^*) = A \mu_y(x^*) + b
            \Sigma(x^*) = A \Sigma_y(x^*) A^T

        where :math:`A` is a rotation matrix, and :math:`b` is a bias vector.

        Args:
            A (np.ndarray[O,O]): square rotation matrix
            b (np.ndarray[O], float, int): bias vector

        Returns:
            KMP: resulting KMP
        """
        kmp = self.multiply(A)
        kmp = kmp.add(b)
        return kmp

    #############
    # Operators #
    #############

    def __str__(self):
        """Return the class name"""
        return self.__class__.__name__

    def __call__(self, x, deterministic=True):
        """Predict output given input data"""
        if deterministic:
            return self.predict(x)
        return self.predict_proba(x)

    def __len__(self):
        """
        Return the length of the database.
        """
        return len(self.database)

    def __iter__(self):
        """
        Iterate over the database and yield the input data with the predicted Gaussian output by GMR.
        """
        for x, gaussian in self.database:
            yield x, gaussian

    def __getitem__(self, index):
        """
        Return the specified entry from the database.

        Args:
            idx (int, slice): index / indices

        Returns:
            np.array: input data
            Gaussian: corresponding predicted Gaussian output (by GMR)
        """
        return self.database[index]

    def __add__(self, bias_vector):
        """
        Add a vector :math:`b` to the prediction of a KMP.

        Args:
            other (np.array[O], float, int): the other vector.

        Returns:
            KMP: resulting KMP
        """
        return self.add(bias_vector)

    def __radd__(self, bias_vector):
        return self.add(bias_vector)

    def __mul__(self, rotation_matrix):
        """
        Rotate the prediction of a KMP by the given rotation matrix.
        Multiply two GMMs, or a GMM by a Gaussian, matrix, or float. See the `multiply` method for more information.

        Warnings: the multiplication of two GMMs performed here is NOT the one that multiply the components
            element-wise. For this one, have a look at `multiply_element_wise` method, or the `__and__` operator.

        Args:
            other (np.array[O,O]): square rotation matrix

        Returns:
            KMP: resulting KMP
        """
        return self.multiply(rotation_matrix)

    def __rmul__(self, rotation_matrix):
        return self.multiply(rotation_matrix)


# TESTS
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create data

    # plot data

    # create KMP model

    # fit/train a KMP

    # predict with KMP

    # plot prediction
    pass
