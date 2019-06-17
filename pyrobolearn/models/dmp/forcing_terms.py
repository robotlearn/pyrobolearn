#!/usr/bin/env python
"""Define the forcing terms used in dynamic movement primitives

This file implements the forcing terms used for discrete and rhythmic dynamic movement primitives.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrobolearn.models.dmp.canonical_systems import *
from pyrobolearn.models.dmp.basis_functions import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ForcingTerm(object):
    r"""Forcing term used in DMPs

    This basically computes the unscaled forcing term, i.e. a weighted sum of basis functions, which is given by:

    .. math:: f(s) = \frac{ sum_{i} \psi_i(s) w_i }{ \sum_i \psi_i(s) }

    where :math:`w` are the learnable weight parameters, and :math:`\psi` are the basis functions evaluated at the
    given input phase variable :math:`s`.
    """

    def __init__(self, weights, basis_functions):
        # check that the arguments have the same length
        self.w = weights
        self.psi = basis_functions

    @property
    def weights(self):
        return self.w

    @staticmethod
    def is_linear():
        return True

    @staticmethod
    def is_parametric():
        return True

    @staticmethod
    def is_recurrent():
        return False

    def compute(self, s):
        """Compute the forcing term

        Compute the value of the forcing term :math:`f(s)` at the given phase value :math:`s`.

        Args:
            s (float): phase value

        Returns:
            float: value of the forcing term at the given phase value
        """
        psi_track = self.psi(s)
        if len(psi_track.shape) == 1:
            return np.dot(psi_track, self.w) / np.sum(psi_track)
        return np.dot(psi_track, self.w) / np.sum(psi_track, axis=1)

    def weighted_basis(self, s):
        """Generate weighted basis

        Returns:
            np.array[T, M]: weighted basis
        """
        return self.psi(s) * self.w

    def normalized_weighted_basis(self, s):
        """Generate normalized weighted basis

        Args:
            s (float): phase value

        Returns:
            np.array[T,M]: normalized weighted basis
        """
        psi_track = self.psi(s)
        return ((psi_track * self.w).T / np.sum(psi_track, axis=1)).T

    # alias
    def __call__(self, s):
        return self.compute(s)

    def __str__(self):
        return self.__class__.__name__

    # To override in child classes
    def train(self, f_target):
        raise NotImplementedError

    # alias
    generate_weights = train


class DiscreteForcingTerm(ForcingTerm):
    r"""Discrete Forcing Term

    .. math:: f(s) = \frac{ sum_{i} \psi_i(s) w_i }{ \sum_i \psi_i(s) } s

    where :math:`w` are the learnable weight parameters, and :math:`\psi` are the basis functions evaluated at the
    given input phase variable :math:`s`.

    This forcing term has the property that as the phase converges to 0, it also converges to 0, allowing the
    linear part of the DMP equation to converge to the goal.
    """

    def __init__(self, cs, num_basis):
        """Initialize the discrete forcing term.

        Args:
            cs (CS): discrete canonical system
            num_basis (int): number of basis functions
        """
        # set canonical system
        if not isinstance(cs, DiscreteCS):
            raise TypeError("Expecting 'cs' to be an instance of DiscreteCS")
        self.cs = cs

        # set num_basis
        self.num_basis = num_basis

        # create weights
        weights = np.zeros(num_basis)  # default f=0

        # desired activations throughout time
        c = np.linspace(0, cs.T, num_basis)
        c = np.exp(-cs.alpha_s * c)

        # set variance of basis functions (this was found by trial and error by DeWolf)
        h = np.ones(num_basis) * num_basis**1.5 / c / cs.alpha_s

        basis = EBF(center=c, h=h)
        super(DiscreteForcingTerm, self).__init__(weights, basis)

    def compute(self, s):
        # call parent compute
        f = super(DiscreteForcingTerm, self).compute(s)
        # scale with phase s
        return f * s

    def train(self, f_target, plot=False):
        """Train the weights to match the given target forcing term

        Generate a set of weights over the basis functions such that the target forcing term trajectory is matched.

        Args:
            f_target (np.array): the desired forcing term trajectory
        """

        # calculate phase and basis functions
        s_track = self.cs.rollout()
        psi_track = self.psi(s_track)   # shape=TxM

        # efficiently calculate BF weights using LWR (Locally Weighted (Linear) Regression)
        # spatial scaling term
        for b in range(self.num_basis):
            numerator = np.sum(s_track * psi_track[:, b] * f_target)
            denominator = np.sum(s_track**2 * psi_track[:, b])
            self.w[b] = numerator / denominator

        self.w = np.nan_to_num(self.w)

        if plot:
            # plot the basis function activations
            plt.figure()
            plt.subplot(211)
            plt.plot(psi_track)
            plt.title('basis functions')

            # plot the desired forcing function vs approx for the first dmp
            plt.subplot(212)
            plt.title('discrete force')
            plt.plot(f_target, label='f_target', linewidth=2.5)
            plt.plot(self.compute(s_track), label='f_pred', linewidth=2.5)

            # weighted sum of basis functions
            wps = self.weighted_basis(s_track)
            plt.plot(wps, linewidth=0.5)

            plt.legend()
            plt.tight_layout()
            plt.show()


class RhythmicForcingTerm(ForcingTerm):
    r"""Rhythmic Forcing Term

    .. math:: f(s) = \frac{ sum_{i} \psi_i(s) w_i }{ \sum_i \psi_i(s) } a

    where :math:`w` are the learnable weight parameters, :math:`\psi` are the basis functions evaluated at the
    given input phase variable :math:`s`, and :math:`a` is the amplitude.

    When used with DMPs, it produces a limit cycle behavior.
    """

    def __init__(self, cs, num_basis, amplitude=1.):
        """
        Initialize the rhythmic forcing term.

        Args:
            cs (CS): rhythmic canonical system
            num_basis (int): number of basis functions
            amplitude (float): amplitude
        """
        # set canonical system
        if not isinstance(cs, RhythmicCS):
            raise TypeError("Expecting 'cs' to be an instance of RhythmicCS")
        self.cs = cs

        # set num_basis and amplitude
        self.num_basis = num_basis
        self.a = amplitude

        # create weights
        weights = np.zeros(num_basis,)  # default f=0

        # set the centre of the Gaussian basis functions to be spaced evenly
        c = np.linspace(0, cs.T, num_basis + 1)  # the '+1' is because it is rhythmic, c(0) = c(2pi)
        c = c[:-1]

        # set concentration of basis function (this was found by trial and error by DeWolf)
        h = np.ones(num_basis) * num_basis

        # create basis functions
        basis = CBF(center=c, h=h)
        super(RhythmicForcingTerm, self).__init__(weights, basis)

    def compute(self, s):
        # call parent compute
        f = super(RhythmicForcingTerm, self).compute(s)
        # scale with amplitude and return it
        return f * self.a

    def train(self, f_target, plot=False):
        """Train the weights to match the given target forcing term

        Generate a set of weights over the basis functions such that the target forcing term trajectory is matched.

        Args:
            f_target (np.array): the desired forcing term trajectory
            plot (bool): If True, it will plot.
        """

        # calculate phase and basis functions
        s_track = self.cs.rollout()
        psi_track = self.psi(s_track)   # shape=TxM

        # efficiently calculate BF weights using LWR (Locally Weighted (Linear) Regression)
        for b in range(self.num_basis):
            self.w[b] = (np.dot(psi_track[:, b], f_target) / (np.sum(psi_track[:, b])))  # + 1e-10))

        if plot:
            # plot the basis function activations
            plt.figure()
            plt.subplot(211)
            plt.plot(psi_track)
            plt.title('basis functions')

            # plot the desired forcing function vs approx for the first dmp
            plt.subplot(212)
            plt.title('rhythmic force')
            plt.plot(f_target, label='f_target', linewidth=2.5)
            plt.plot(self.compute(s_track), label='f_pred', linewidth=2.5)
            wps = self.weighted_basis(s_track)
            plt.plot(wps, linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()


# Tests
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # tests canonical systems
    discrete_cs = DiscreteCS()
    rhythmic_cs = RhythmicCS()

    # plot canonical systems
    plt.subplot(1, 2, 1)
    plt.title('Discrete CS')
    for tau in [1., 0.5, 2.]:
        rollout = discrete_cs.rollout(tau=tau)
        plt.plot(np.linspace(0, 1., len(rollout)), rollout, label='tau='+str(tau))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Rhythmic CS')
    for tau in [1., 0.5, 2.]:
        rollout = rhythmic_cs.rollout(tau=tau)
        plt.plot(np.linspace(0, 1., len(rollout)), rollout, label='tau='+str(tau))
    plt.legend()
    plt.show()

    # tests basis functions
    num_basis = 20
    discrete_f = DiscreteForcingTerm(discrete_cs, num_basis)
    rhythmic_f = RhythmicForcingTerm(rhythmic_cs, num_basis)

    plt.subplot(1, 2, 1)
    rollout = discrete_cs.rollout()
    plt.title('discrete basis fcts')
    plt.plot(rollout, discrete_f.psi(rollout))

    plt.subplot(1, 2, 2)
    rollout = rhythmic_cs.rollout()
    plt.title('rhythmic basis fcts')
    plt.plot(rollout, rhythmic_f.psi(rollout))
    plt.show()

    # tests forcing terms
    f = np.sin(np.linspace(0, 2*np.pi, 100))
    discrete_f.train(f, plot=True)

    f = np.sin(np.linspace(0, 2*np.pi, int(2*np.pi*100)))
    rhythmic_f.train(f, plot=True)
