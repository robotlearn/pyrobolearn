#!/usr/bin/env python
"""Provide the Principal Component Analysis model.

Dependencies: None
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PCA(object):
    r"""Principal Component Analysis (PCA)

    This class describes PCA; a non-parametric, linear model which possesses the 3 following properties [1]:
    * linearity
    * orthogonality
    * high signal-noise ratio

    PCA can be achieved by performing Singular Value Decomposition (SVD) on the data, or performing an
    eigendecomposition on the covariance data matrix.

    Assuming the mean-centered data matrix is given by :math:`X \in \mathbb{R}^{N \times M}`, applying SVD on it
    gives us:
    .. math:: X = USV^T,
    where :math:`U \in \mathbb{R}^{N \times N}` is an orthogonal matrix where its columns represent the eigenvectors
    of :math:`XX^T` also known as the left-singular vectors of :math:`X`, :math:`S \in \mathbb{R}^{N \times M}`
    contains the singular values ordered by descending order, and :math:`V \in \mathbb{R}^{M \times M}` is an
    orthogonal matrix in which its columns represent the eigenvectors of :math:`X^TX` also known as the right-singular
    vectors of :math:`X`. The columns of :math:`V` form a basis spanning :math:`\mathbb{R}^M`.

    Applying eigendecomposition on the covariance matrix :math:`C_X \sim X^TX` gives us:
    .. math:: X^TX = QLQ^T.

    While applying SVD on :math:`X` and then computing the covariance gives us:
    .. math:: X^TX = (USV^T)^T (USV^T) = VSSV^T = VS^2V^T

    Thus, :math:`Q=V` (i.e. same orthogonal matrix containing the evecs) and :math:`L=S^2` (that is, the eigenvalues
    are the square of the singular values). Note that the covariance :math:`C_X` is formally defined as
    :math:`\frac{1}{(N-1)} X^TX` where :math:`N` is the number of data points, and not :math:`X^TX`, then
    :math:`L=\frac{S^2}{(N-1)}`.

    PCA can be formulated as an optimization process, which consists to find the dimensions that maximize
    the projected variance. Specifically, this can be written as:

    .. math::
        max_{u_i} ||Xu_i||^2  & \mbox{ subj. to } u_i^Tu_i = 1; u_j^Tu_i = 0  \\
        max_{u_i} (Xu_i)^T(Xu_i) & \mbox{ subj. to } u_i^Tu_i = 1; u_j^Tu_i = 0 \\
        max_{u_i} u_i^TX^TXu_i & \mbox{ subj. to } u_i^Tu_i = 1; u_j^Tu_i = 0 \\

    :math:`\forall i, \forall j < i`. The solution is given by the column vectors of the matrix :math:`V`, that is,
    the eigenvectors, while the obtained values during the maximization process are given by the eigenvalues.
    This is why PCA can be performed by applying SVD (on the data matrix) or Eigendecomposition (on the covariance
    matrix).

    Complexity:
        * Spatial complexity: O(NM)
        * Time complexity: O(min(NM^2, MN^2))

    References:
        [1] "A Tutorial on Principal Component Analysis", Shlens, 2014
    """
    def __init__(self, X=None, normalize_data=False):
        if X is not None:
            self.train(X, normalize=normalize_data)

        self.evals, self.evecs = None, None

    ##############
    # Properties #
    ##############

    @property
    def eigenvalues(self): # alias to evals
        return self.evals

    @property
    def eigenvectors(self): # alias to evecs
        return self.evecs

    ##################
    # Static Methods #
    ##################

    # TODO: think if PCA can be considered as a model

    @staticmethod
    def is_parametric():
        """PCA is a non-parametric approach"""
        return False

    @staticmethod
    def is_linear():
        """PCA does not have parameters, but it is a linear dimensionality reduction algo"""
        return True

    @staticmethod
    def is_recurrent():
        """PCA is not recurrent"""
        return False

    @staticmethod
    def is_latent():
        """PCA gives a latent model"""
        return True

    @staticmethod
    def is_probabilistic():
        """PCA is not a probabilistic approach but a deterministic one"""
        return False

    @staticmethod
    def is_discriminative():
        """PCA is a discriminative model, which projects the given data into a lower space"""
        return True

    @staticmethod
    def is_generative():
        """PCA is not a generative model from which you can sample from it"""
        return False

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Return an iterator over the parameters."""
        raise RuntimeError("PCA doesn't have any parameters.")

    def named_parameters(self):
        """Return an iterator over the parameters, yielding both the name and the parameter itself."""
        raise RuntimeError("PCA doesn't have any parameters.")

    def list_parameters(self):
        """Return the list of parameters."""
        return list(self.parameters())

    def hyperparameters(self):
        """Return an iterator over the hyper-parameters."""
        pass

    def named_hyperparameters(self):
        """Return an iterator over the hyper-parameters, yielding both the name and the hyper-parameter itself."""
        pass

    def list_hyperparameters(self):
        """Return the list of hyper-parameters."""
        return list(self.hyperparameters())

    def train(self, X, normalize=False, copy=True):
        """
        Compute PCA on the given data. This method will center the data.

        Args:
            X (float[N,D]): the matrix to apply PCA on (with shape(X)=NxD, where `N` is the nb of samples, and
                `D` is the dimensionality of a sample).
            normalize (bool): if True, it will normalize the data using the std dev. PCA will then be applied on
                the correlation matrix instead of the covariance matrix.
            copy (bool): if True, it will first copy the data before centering it, and possibly normalizing it.

        Return:
            float[D]: the sorted eigenvalues
            float[D]: the sorted eigenvectors
        """
        if copy:
            X = np.copy(X)

        # 1. Center the data
        mean = X.mean(axis=0)
        X -= mean
        N = X.shape[0]

        # Normalize using the std dev
        if normalize:
            X /= X.std(axis=0)

        # 2. Compute the covariance/correlation matrix
        CovX = 1./(N-1) * X.T.dot(X) # TxT (same as np.cov(X, rowvar=False)))

        # 3. Compute the eigenvectors of this covariance matrix
        # np.linalg.eigh is more efficient than np.linalg.eig for symmetric matrix
        evals, evecs = np.linalg.eigh(CovX)

        # 4. Sort the eigenvalues (in decreasing order) and eigenvectors
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:,idx]

        # save values
        self.evals = evals
        self.evecs = evecs

        return evals, evecs

    def predict(self, x):
        # TODO
        pass


class RecursivePCA(PCA):
    r"""Recursive PCA

    Given a new data point, the

    Assume the covariance matrix is given by :math:`C`, and the mean by :math:`m`, then the new covariance matrix
    accounting for the new data point is given by:

    .. math:: C' = C + \frac{N}{N+1} (m m^T - m x'^T - x' m^T + x'x'^T)

    where :math:`C` can be reconstructed from the eigenvalues and eigenvectors using the eigendecomposition:

    .. math:: C = V \Lambda V^T

    and the new mean is given by:

    .. math::

    Finally, the total number of data points N is updated. This reduces the the spatial and time complexities to
    O(T^2) and O(T^3), respectively.

    Time complexity:
    * O(min(NT^2, TN^2))
    """

    def __init__(self, X=None, normalize_data=False):
        super(RecursivePCA, self).__init__(X, normalize_data)

    def train_recursive(self, X):
        if self.X is None:
            # apply std PCA
            self.X = self.train(X)
            self.mean = np.mean(X, axis=0).reshape(-1, 1)
            self.cov = np.cov(X, rowvar=False)
            self.N = len(X)

        else:  # recursive
            for x in X:
                x = x.reshape(-1, 1)
                Y = self.mean.dot(x.T)
                self.cov = self.cov + self.N / (self.N + 1.) * (self.mean.dot(self.mean.T) - Y - Y.T
                                                            + self.mean.dot(self.mean.T))
                self.mean = self.N / (self.N + 1.) * self.mean + 1 / (N+1) * x

                self.N += 1


class HierarchicalPCA(PCA):
    r"""Hierarchical PCA

    Assume :math:`X = [X_1, X_2] \in \mathbb{R}^{N \times 2D}`, where :math:`X_1, X_2 \in \mathbb{R \times D}`.

    We first decompose the covariance matrix :math:`C` of :math:`X` in terms of :math:`X_1` and :math:`X_2`, and
    apply SVD on each term as follows:

    .. math::

        C = X^T X
        = \left[ \begin{array}{cc}
            X_1^T X_1 & X_1^T X_2 \\
            X_2^T X_1 & X_2^T X_2
        \end{array} \right]
        = \left[ \begin{array}{cc}
            V_1 \Lambda_1 V_1^T & V_1 \Sigma_1 U_1^T U_2 \Sigma_2 V_2^T \\
             V_2 \Sigma_2 U_2^T U_1 \Sigma_1 V_1^T & V_2 \Lambda_2 V_2^T
        \end{array} \right]

    Similarly, we apply the eigendecomposition on C which results in:

    .. math::

        C = X^T X = V \Lambda V^T
        = \left[ \begin{array}{cc} V_{11} & V_{12} \\ V_{21} & V_{22} \end{array} \right]
        \left[ \begin{array}{cc} \Lambda_{11} & 0 \\ 0 & \Lambda_{22} \end{array} \right]
        \left[ \begin{array}{cc} V_{11}^T & V_{21}^T \\ V_{12}^T & V_{22}^T \end{array} \right]
        = \left[ \begin{array}{cc}
            V_{11} \Lambda_{11} V_{11}^T + V_{12} \Lambda_{22} V_{12}^T
            & V_{11} \Lambda_{11} V_{21}^T + V_{12} \Lambda_{22} V_{22}^T \\
            V_{21} \Lambda_{11} V_{11}^T + V_{22} \Lambda_{22} V_{12}^T
            & V_{21} \Lambda_{11} V_{21}^T + V_{22} \Lambda_{22} V_{22}^T
        \end{array} \right]

    """
    pass
