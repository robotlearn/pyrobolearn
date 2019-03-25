#!/usr/bin/env python
"""Define canonical systems for dynamic movement primitives

This file implements canonical systems for discrete and rhythmic dynamic movement primitives.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CS(object):
    r"""Canonical System.

    A canonical system (CS) drives a dynamic movement primitive (DMP) by providing a phase variable [1].
    The phase variable was introduced to avoid an explicit dependency with time in the DMP equations. Canonical
    systems can be categorized in two main categories:
    * discrete CS: used for discrete movements (such as reaching, pushing/pulling, hitting, etc)
    * rhythmic CS: used for rhythmic movements (such as walking, running, dribbling, sewing, flipping a pancake, etc)

    Each of these systems are described by differential equations which are solved using Euler's method.
    See their corresponding classes `DiscreteCS` and `RhythmicCS` for more information.

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    __metaclass__ = ABCMeta

    def __init__(self, dt=0.01, T=1.):
        """Initialize the canonical system.

        Args:
            dt (float): the time step used in Euler's method when solving the differential equation
                A very small step will lead to a better accuracy but will take more time.
        """
        # set variables
        self.dt = dt
        self.T = T
        self.timesteps = int(T / self.dt)
        # rescale integration step (same as np.linspace(0.,T.,timesteps) instead of np.arange(0,T,dt))
        self.dt = self.T / (self.timesteps - 1.)

        self.init_phase = 1.0
        self.s = 1.0

        # reset the phase variable
        self.reset()

    @abstractmethod
    def step(self, tau=1.0, error_coupling=1.0):
        """Perform a step using Euler's method. This needs to be implemented in the child classes."""
        raise NotImplementedError

    def reset(self):
        """Reset the phase variable"""
        self.s = self.init_phase
        return self.s

    def rollout(self, tau=1.0, error_coupling=1.0):
        """Generate phase variable in an open loop fashion.

        Args:
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            error_coupling (float): slow down if the error is > 1
        """
        timesteps = int(self.timesteps * tau)
        self.s_track = np.zeros(timesteps)

        # reset
        self.reset()

        # roll
        for t in range(timesteps):
            self.s_track[t] = self.s
            self.step(tau, error_coupling)

        return self.s_track


class DiscreteCS(CS):
    r"""Discrete Canonical System.

    The discrete canonical system drives the various DMPs by providing the phase variable at each time step, and is
    given by:

    .. math:: \tau \dot{s} = - \alpha_s s

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the movement, :math:`s` is the phase
    variable that drives the DMP, and :math:`\alpha_s` is a predefined constant.
    This differential equation is solved using Euler's method.

    This version is used for discrete movements, where :math:`s` starts from 1 and converge to 0 as time progresses.
    The phase variable was introduced to avoid an explicit dependency of time in the DMP equations.

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
    """

    def __init__(self, alpha_s=1, dt=0.01):
        super(DiscreteCS, self).__init__(dt=dt, T=1.0)
        self.alpha_s = alpha_s

    def reset(self):
        """Reset the phase variable"""
        self.s = self.init_phase
        return self.s

    def step(self, tau=1.0, error_coupling=1.0):
        """Generate phase value for discrete movements.

        The phase variable :math:`s` is generated by solving :math:`\tau \dot{s} = - \alpha_s s` using Euler's method.
        This phase decays from 1 to 0.

        Args:
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            error_coupling (float): slow down if the error is > 1

        Returns:
            float: phase value
        """
        s = self.s
        self.s += (-self.alpha_s/tau * self.s * error_coupling) * self.dt
        # return self.s
        return s


class RhythmicCS(CS):
    r"""Rhythmic Canonical System.

    The rhythmic canonical system drives the various DMPs by providing a phase variable that is periodic [1]. It is
    used for rhythmic movements (such as walking, dribbling, sewing, etc.) and is given by:

    .. math:: \tau \dot{s} = 1

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the movement, :math:`s` is the phase
    variable that drives the DMP. This differential equation is solved using Euler's method.

    Rhythmic canonical systems can also be coupled with each other as done in [2] to synchronize various DMPs.

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
        [2] "A Framework for Learning Biped Locomotion with Dynamical Movement Primitives", Nakanishi et al., 2004
    """

    def __init__(self, dt=0.01):
        super(RhythmicCS, self).__init__(dt=dt, T=2*np.pi)
        self.init_phase = 0.0

    def reset(self):
        """Reset the phase variable"""
        self.s = self.init_phase
        return self.s

    def step(self, tau=1.0, error_coupling=1.0):
        r"""Generate phase value for rhythmic movements.

        The phase variable :math:`s` is generated by solving :math:`\tau \dot{s} = 1` using Euler's method.

        Args:
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            error_coupling (float): slow down if the error is > 1

        Returns:
            float: phase value
        """
        s = self.s
        self.s += (1./tau * error_coupling) * self.dt
        # return self.s
        return s


class RhythmicNetworkCS(CS):
    r"""Rhythmic Network CS.

    In this version, instead of having one canonical system that drives all the various DMPs, we have several
    canonical systems coupled with each other, and where each one of them is associated to a particular DMP.

    The evolution of the phase variable :math:`\phi` of the system :math:`i` is given by:

    .. math:: \dot{\phi}_i = \omega_i + \sum_j a_j w_{ij} \sin(\phi_j - \phi_i - \varphi_{ij})

    where :math:`\omega` is the desired angular velocity (desired frequency), :math:`w_{ij}` are the coupling weights,
    :math:`\varphi_{ij}` are the phase biases, and :math:`a_j` are the amplitudes of the other systems :math:`j`.
    This formulation is similar to Central Pattern Generators (CPGs), see [3].

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
        [2] "A Framework for Learning Biped Locomotion with Dynamical Movement Primitives", Nakanishi et al., 2004
        [3] "Central pattern generators for locomotion control in animals and robots: a review", Ijspeert, 2008
    """
    def __init__(self, dt=0.01):
        super(RhythmicNetworkCS, self).__init__(dt=dt)


# Tests
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # tests canonical systems
    discrete_cs = DiscreteCS()
    rhythmic_cs = RhythmicCS()

    # check tau
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
