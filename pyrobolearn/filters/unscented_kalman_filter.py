#!/usr/bin/env python
"""Define the Unscented Kalman Filter.
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


class UKF(object):
    r"""Unscented Kalman Filter (UKF)

    Type: Derivative-free Gaussian filter (=Bayes filter for continuous spaces)

    This is an extension and generalization to the standard Kalman filter which also works with nonlinear systems.
    This filter is an alternative to deal with non-linear process/measurement models, using :math:`\sigma`-points
    to approximate the probability distribution. Specifically, as EKF, it linearizes the transformation of a Gaussian,
    but instead of doing it via a Taylor expansion, it "performs a stochastic linearization through the use of
    a weighted statistical linear regression process". [1]

    The dynamic model :math:`p(x_t | x_{t-1}, u_t)` and the measurement model :math:`p(z_t | x_t)` are given by:

    .. math::

        x_t = f(x_{t-1}, u_t) + \epsilon_t \qquad \mbox{where} \qquad \epsilon_t \sim \mathcal{N}(0, R_t)
        z_t = h(x_t) + \delta_t \qquad \mbox{where} \qquad \delta_t \sim \mathcal{N}(0, Q_t)

    where :math:`f` and :math:`h` are nonlinear functions.

    The UKF works by generating :math:`2n+1` :math:`sigma`-points, and pass them through the corresponding dynamical
    and measurement functions, and compute from them a weighted empirical mean and covariance.

    Notes:
        - Compared to EKF, this filter does not require to compute the Jacobians for the dynamical and measurement
        models, and is thus a derivative-free filter. "If the belief is highly non-Gaussian, then the UKF
        representation is too restrictive and the filter can perform arbitrarily poorly" [1]
        - An extension to multimodal posteriors is known as multi-hypothesis Kalman filter which uses a mixture of
        Gaussians. See also, the histogram and particle filters that are well-suited to represent complex multimodal
        beliefs. [1]

    Complexity: "The asympototic complexity of the UKF algorithm is the same as for the EKF. In practice, the EKF is
        often slightly faster than the UKF. Nevertheless, the UKF is still highly efficient" [1]

    References:
        [1] "Probabilistic Robotics", Thrun et al., 2006 (sec 3.4)
    """

    def __init__(self, mean, covariance, dynamic_noise_cov, measurement_noise_cov, alpha=1., kappa=0, beta=2):
        """
        Initialize the Unscented Kalman Filter.

        Args:
            mean (float[N]): initial mean of the state belief
            covariance (float[N,N]): initial covariance matrix of the state belief
            dynamic_noise_cov (float[N,N]): dynamic noise covariance matrix
            measurement_noise_cov (float[K,K]): measurement noise covariance matrix
            alpha (float): control the spread of the sigma points (recommended value: 1e-3 or 1)
            kappa (float): control the spread of the sigma points
            beta (int): this parameter can be chosen to encode additional (higher order) knowledge about the
                distribution underlying the Gaussian representation.
        """
        self.mu = mean
        self.Sigma = covariance
        self.n = self.mu.size
        self.tau = alpha**2 * (self.n + kappa) - self.n
        self.weight_mean_0 = self.tau / (self.n + self.tau)
        self.weight_cov_0 = self.tau / (self.n + self.tau) + (1 - alpha**2 + beta)
        self.weight = 1./(2.*(self.n + self.tau))
        self.gamma = np.sqrt(self.n + self.tau)

    def generate_sigma_points(self):
        """
        Generate the sigma points for the UKF.

        Returns:
            float[2N+1, N]: sigma points
        """
        term = self.gamma * np.sqrt(self.Sigma.T)
        sigma_points = np.vstack((self.mu, self.mu + term, self.mu - term))
        return sigma_points

    def predict(self, f, u):
        """
        Predict (a priori) the next state (without taking into account measurements) using the nonlinear dynamic model
        with added Gaussian noise.

        Args:
            f (callable class/function): (nonlinear) dynamical function.
            u (np.array): control array

        Returns:
            float[N]: a priori mean of the Gaussian belief
            float[N,N]: a priori covariance of the Gaussian belief
        """
        # generate sigma points
        sigma_points = self.generate_sigma_points()

        # feed each sigma point to the dynamical function
        X = np.array([f(sigma, u) for sigma in sigma_points])

        # compute the weighted empirical mean and covariance
        self.mu = self.weight_mean_0 * X[0] + self.weight * X[1:].sum(axis=0)
        diff0 = (X[0] - self.mu).reshape(-1,1)
        diff = (X[1:] - self.mu).T
        self.Sigma = self.weight_cov_0 * diff0.dot(diff0.T) + self.weight * diff.dot(diff.T) + self.R

        # return the a priori Gaussian belief on the next state
        return self.mu, self.Sigma

    def measurement_update(self, h, z):
        """
        Predict (a posteriori) the next state by incorporating the measurement :math:`z_t`.

        Args:
            h (callable class/function): (nonlinear) measurement function.
            z (np.array): measurement array

        Returns:
            float[N]: a posteriori mean of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        # generate sigma points
        sigma_points = self.generate_sigma_points()

        # feed each sigma point to the measurement function
        Z = np.array([h(sigma) for sigma in sigma_points])

        # weighted empirical mean for the measurement
        z_mean = self.weight_mean_0 * Z[0] + self.weight * Z[1:].sum(axis=0)

        # deviation error between the measurement and the a priori predicted state (aka innovation)
        y = z - z_mean

        # compute the weighted cross-covariance
        term0 = ((sigma_points[0] - self.mu).reshape(-1,1)).dot( (Z[0] - z_mean).reshape(1, -1) )
        term = ((sigma_points[1:] - self.mu).T).dot(Z[1:] - z_mean)
        C = self.weight_cov_0 * term0 + self.weight * term

        # Innovation covariance matrix
        diff0 = (Z[0] - z_mean).reshape(-1, 1)
        diff = (Z[1:] - z_mean).T
        S = self.weight_cov_0 * diff0.dot(diff0.T) + self.weight * diff.dot(diff.T) + self.Q

        # Kalman gain (which specifies the degree to which the measurement is incorporated into the new state)
        K = C.dot(np.linalg.inv(S))

        # compute the weighted empirical mean and covariance
        self.mu = self.mu + K.dot(y)
        self.Sigma = self.Sigma - K.dot(S.dot(K.T))

        # return the a posteriori Gaussian belief on the next state
        return self.mu, self.Sigma

    def compute(self, f, u, h, z):
        """
        Perform a prediction and measurement update step.

        Args:
            f (callable class/function): (nonlinear) dynamical function.
            u (np.array): control array
            h (callable class/function): (nonlinear) measurement function.
            z (np.array): measurement array

        Returns:
            float[N]: a posteriori mean vector of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        self.predict(f, u)
        self.measurement_update(h, z)
        return self.mu, self.Sigma
