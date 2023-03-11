#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Extented Kalman Filter.
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class EKF(object):
    r"""Extended Kalman Filter (EKF)

    Type: Gaussian filter (=Bayes filter for continuous spaces)

    This is an extension and generalization to the standard Kalman filter which also works with nonlinear systems.
    This is more practical as state transitions and measurements are seldomly linear. The EKF calculates a Gaussian
    distribution to the true belief, and represents thus an approximate of the belief (i.e. posterior).

    The dynamic model :math:`p(x_t | x_{t-1}, u_t)` and the measurement model :math:`p(z_t | x_t)` are given by:

    .. math::

        x_t = f(x_{t-1}, u_t) + \epsilon_t \qquad \mbox{where} \qquad \epsilon_t \sim \mathcal{N}(0, R_t)
        z_t = h(x_t) + \delta_t \qquad \mbox{where} \qquad \delta_t \sim \mathcal{N}(0, Q_t)

    where :math:`f` and :math:`h` are nonlinear functions.

    Because EKF cannot compute the true statistics in closed form, it uses linearization (via Taylor expansion).
    That is, it approximates :math:`f` and :math:`h` by a linear function. Note that another way would be to compute
    a Monte-carlo estimate of the Gaussian but would require a higher time-complexity (dependent on the number of
    samples).

    In a mathematical language, the linearization via Taylor expansion is given by:

    .. math::

        f(x_{t-1}, u_t) = f(\mu_{t-1}, u_t) + f'(\mu_{t-1}, u_t) (x_{t-1} - \mu_{t-1})
        h(x_t) = h(\mu_t) + h'(\mu_t) (x_t - \mu_t)

    where :math:`f'(\mu_{t-1}, u_t)` and :math:` h'(\mu_t)` represent the Jacobians evaluated at the means.

    Notes: EKF requires to compute the Jacobians for the dynamical and measurement models.

    Complexity: :math:`O(d^3 + n^2)` because of the matrix inversion when computing the kalman gain. :math:`d` is the
    dimension of the measurement vector, and :math:`n` is the dimension of the state space.

    References:
        [1] "Probabilistic Robotics", Thrun et al., 2006 (sec 3.3)
    """

    def __init__(self, mean, covariance, dynamic_noise_cov, measurement_noise_cov):
        """
        Initialize the Extended Kalman Filter.

        Args:
            mean (float[N]): initial mean of the state belief
            covariance (float[N,N]): initial covariance matrix of the state belief
            dynamic_noise_cov (float[N,N]): dynamic noise covariance matrix
            measurement_noise_cov (float[K,K]): measurement noise covariance matrix
        """
        self.mu = mean
        self.Sigma = covariance
        self.R = dynamic_noise_cov
        self.Q = measurement_noise_cov
        self.I = np.identity(self.Sigma.shape[0])

    def predict(self, f, u):
        """
        Predict (a priori) the next state (without taking into account measurements) using the nonlinear dynamic model
        with added Gaussian noise.

        Args:
            f (callable class): (nonlinear) dynamical function. This should have a method `jacobian`.
            u (np.array): control array

        Returns:
            float[N]: a priori mean of the Gaussian belief
            float[N,N]: a priori covariance of the Gaussian belief
        """
        F = f.jacobian(self.mu, u)

        # predict a priori the next state (by only using the nonlinear dynamic model)
        self.mu = f(self.mu, u)
        self.Sigma = F.dot(self.Sigma).dot(F.T) + self.R

        # return the a priori Gaussian belief on the next state
        return self.mu, self.Sigma

    def measurement_update(self, h, z):
        """
        Predict (a posteriori) the next state by incorporating the measurement :math:`z_t`.

        Args:
            h (callable class): (nonlinear) measurement function. This should have a method `jacobian`.
            z (np.array): measurement array

        Returns:
            float[N]: a posteriori mean of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        H = h.jacobian(self.mu)

        # deviation error between the measurement and the a priori predicted state (aka innovation)
        y = z - h(self.mu)

        # Innovation covariance matrix
        S = H.dot(self.Sigma).dot(H.T) + self.Q

        # Kalman gain (which specifies the degree to which the measurement is incorporated into the new state)
        K = self.Sigma.dot(H.T).dot(np.linalg.inv(S))

        # predict a posteriori the next state
        self.mu = self.mu + K.dot(y)
        self.Sigma = (self.I - K.dot(H)).dot(self.Sigma)

        # return the a posteriori Gaussian belief on the next state
        return self.mu, self.Sigma

    def compute(self, f, u, h, z):
        """
        Perform a prediction and measurement update step.

        Args:
            f (callable class): (nonlinear) dynamical function. This should have a method `jacobian`.
            u (np.array): control array
            h (callable class): (nonlinear) measurement function. This should have a method `jacobian`.
            z (np.array): measurement array

        Returns:
            float[N]: a posteriori mean vector of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        self.predict(f, u)
        self.measurement_update(h, z)
        return self.mu, self.Sigma
