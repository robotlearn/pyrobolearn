#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Kalman Filter state estimator.
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


class KalmanFilter(object):
    r"""Kalman Filter (KF)

    Type: Gaussian filter (=Bayes filter for continuous spaces)

    The Kalman filter (also known as the linear quadratic estimator (LQE)) is a state estimator which "uses a series
    of measurements observed over time, containing statistical noise and other inaccuracies, and produces estimates
    of unknown variables that tend to be more accurate than those based on a single measurement alone, by estimating
    a joint probability distribution over the variables for each timeframe." (Wikipedia)

    Assumptions:
        * Markov assumption
        * linear dynamic model with added Gaussian noise
        * linear measurement model with added Gaussian noise
        * initial belief on the state is normally distributed

    The Kalman filter consists of two steps:
        * Prediction step: produces the a priori estimate of the current state variables, along with their
            corresponding uncertainties, based on the dynamical model.
        * Measurement update step: produces the a posteriori estimate of the state along with its uncertainty,
            by taking into account the received measurements, comparing and incorporating them with the above
            prediction.

    The dynamic model :math:`p(x_t | x_{t-1}, u_t)` and the measurement model :math:`p(z_t | x_t)` are given by:

    .. math::

            x_t = A_t x_{t-1} + B_t u_t + \epsilon_t \qquad \mbox{where} \qquad \epsilon_t \sim \mathcal{N}(0, R_t)
            z_t = C_t x_t + \delta_t \qquad \mbox{where} \qquad \delta_t \sim \mathcal{N}(0, Q_t)

    Notes: The use of the KF " does not assume that the errors are Gaussian. However, the filter yields the exact
    conditional probability estimate in the special case that all errors are Gaussian." (Wikipedia)

    Complexity: :math:`O(d^3 + n^2)` because of the matrix inversion when computing the kalman gain. :math:`d` is the
    dimension of the measurement vector, and :math:`n` is the dimension of the state space.

    References:
        [1] "A New Approach to Linear Filtering and Prediction Problems", Kalman, 1960
        [2] "Probabilistic Robotics", Thrun et al., 2006 (sec 3.2)
    """

    def __init__(self, mean, covariance, dynamic_noise_cov, measurement_noise_cov):
        """
        Initialize the Kalman filter.

        Args:
            mean (float[N]): initial mean of the state belief
            covariance (float[N,N]): initial covariance matrix of the state belief
            dynamic_noise_cov (float[N,N]): dynamic noise covariance matrix
            measurement_noise_cov (float[K,K]): measurement noise covariance matrix
        """
        # self.belief = Gaussian(...)
        self.mu = mean
        self.Sigma = covariance
        self.R = dynamic_noise_cov
        self.Q = measurement_noise_cov
        self.I = np.identity(self.Sigma.shape[0])

    def predict(self, A, B, u):
        """
        Predict (a priori) the next state (without taking into account measurements) using the linear dynamic model
        with added Gaussian noise.

        Args:
            A (float[N,N]): linear state transformation matrix
            B (float[N,M]): linear control transformation matrix
            u (float[M]): control vector

        Returns:
            float[N]: a priori mean of the Gaussian belief
            float[N,N]: a priori covariance of the Gaussian belief
        """
        # predict a priori the next state (by only using the linear dynamic model)
        self.mu = A.dot(self.mu) + B.dot(u)
        self.Sigma = A.dot(self.Sigma).dot(A.T) + self.R

        # return the a priori Gaussian belief on the next state
        return self.mu, self.Sigma

    def measurement_update(self, C, z):
        """
        Predict (a posteriori) the next state by incorporating the measurement :math:`z_t`.

        Args:
            C (float[K,N]): linear state-measurement transformation matrix
            z (float[K]): measurement vector

        Returns:
            float[N]: a posteriori mean vector of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        # deviation error between the measurement and the a priori predicted state (aka innovation)
        y = z - C.dot(self.mu)

        # Innovation covariance matrix
        S = C.dot(self.Sigma).dot(C.T) + self.Q

        # Kalman gain (which specifies the degree to which the measurement is incorporated into the new state)
        K = self.Sigma.dot(C.T).dot(np.linalg.inv(S))

        # predict a posteriori the next state
        self.mu = self.mu + K.dot(y)
        self.Sigma = (self.I - K.dot(C)).dot(self.Sigma)

        # return the a posteriori Gaussian belief on the next state
        return self.mu, self.Sigma

    def compute(self, A, B, u, C, z):
        """
        Perform a prediction and measurement update step.

        Args:
            A (float[N,N]): linear state transformation matrix
            B (float[N,M]): linear control transformation matrix
            u (float[M]): control vector
            C (float[K,N]): linear state-measurement transformation matrix
            z (float[K]): measurement vector

        Returns:
            float[N]: a posteriori mean vector of the Gaussian belief
            float[N,N]: a posteriori covariance matrix of the Gaussian belief
        """
        self.predict(A, B, u)
        self.measurement_update(C, z)
        return self.mu, self.Sigma
