#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the general dynamic movement primitive abstract class.

This file implements the DMP abstract class from which all dynamic movement primitive classes inherit from.
"""

import numpy as np
import copy
import scipy.interpolate
import matplotlib.pyplot as plt

from pyrobolearn.models.dmp.canonical_systems import CS
from pyrobolearn.models.dmp.forcing_terms import ForcingTerm


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DMP(object):
    r"""Dynamic Movement Primitive

    Dynamic movement primitives (DMPs) are a set of differential equations (for each degree of freedoms (DoFs), i.e.
    general coordinates) that encodes a movement [1]. It is thought that movement primitives are the building blocks
    of a movement, and several evidences show that such modules exist in animals [2].

    DMPs are often formulated as a 2nd-order differential equation:

    .. math:: \tau^2 \ddot{y} = \alpha ( \beta (g - y) - \dot{y}) + f(s)

    or sometimes, as a first-order differential system:

    .. math::

            \tau \dot{z} &= \alpha ( \beta (g - y) - z) + f(s) \\
            \tau \dot{y} &= z

    They can also be rewritten as:

    .. math:: \tau^2 \ddot{y} = K (g - y) - D \tau \dot{y} + f(s)

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the reproduced movement, :math:`K`
    is the stiffness coefficient, :math:`D` is the damping coefficient, :math:`y, \dot{y}, \ddot{y}` are the position,
    velocity, and acceleration of a DoF, and :math:`f(s)` is the non-linear forcing term. These equations are also
    known as the transformation systems and represent, with the canonical system, DMPs.

    All of the above formulations are equivalent to each other. However, in my humble opinion, the last equation
    depicts better what the transformation system constitutes; it is a unit-mass spring-damper system or PD controller
    with a forcing term. This last term is non-linear and can be learned from the demonstrations.
    If the forcing is zero, then the differential equation is stable, and the position :math:`y` converges to the goal.
    The stiffness and damping coefficients (:math:`K` and :math:`D`) are often selected such that the whole system
    (without the forcing term) is critically damped (:math:`D = 2 \sqrt{K}`). Other behaviors can be obtained by
    selecting the stiffness and damping coefficient such that we obtain:
    * an undamped system: :math:`D = 0` or :math:`K \rightarrow \infty`
    * an underdamped system: :math:`D < 2 \sqrt{K}`
    * a critically damped system: :math:`D = 2 \sqrt{K}`
    * an overdamped system: :math:`D > 2 \sqrt{K}`

    Because the last formulation is more intuitive (at least for me), it will be used in this class.
    Imitation is performed by learning the forcing term.

    DMPs can be categorized in two main categories:
    * discrete DMP: used to represent discrete movements such as such as reaching, pushing/pulling, etc.
    * rhythmic DMP: used to represent rhythmic movements such as walking, running dribbling, sewing, etc.

    DMP have the following nice properties:
    * translation invariant
    * linear parameters but still allows to represent non-linear movements

    Here are few limitations/shortcomings:
    * hard to couple sensory information with it
    * have to come up with the number of basis functions

    For a more biologically-inspired DMP [5] which allows to adapt the goal in real-time and a better rescaling, see
    the `BioDMP` class.

    Note that this code was inspired by the `pydmps` code [2,3], but differ in several ways, notably:
    - we undertake a more object-oriented programming (OOP) approach
    - the equations are a little bit different (e.g. :math:`tau`) in which we use the ones presented in the refs
    - we decouple the Euler's method time step with the time step for the number of data points
    - timesteps: we go from 0 to T included, while DeWolf goes from 0 to T-1
    - we use array operation instead of iterating over each element to update them
    - we enforce consistency between the various methods and data structures
    - we implement `BioDMP` which allows to adapt and rescale the goal in real-time based on [4]
    - we implemented DMP sequencing based on [5]
    - we implemented DMP that can be used with orientations based on [7]
    - phase nodes which allows to couple phases, such as done in [8] for locomotion
    - it can be used with RL algorithms, notably PoWER [9] and PI^2 [10]

    References:
        - [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
        - [2] "Motor primitives in vertebrates and invertebrates", Flash et al., 2005
        - [3] Tutorials on DMP: https://studywolf.wordpress.com/category/robotics/dynamic-movement-primitive/
        - [4] PyDMPs (from DeWolf, 2013): https://github.com/studywolf/pydmps
        - [5] "Biologically-inspired Dynamical Systems for Movement Generation: Automatic Real-time Goal Adaptation
          and Obstacle Avoidance", Hoffmann et al., 2009
        - [6] "Action Sequencing using Dynamic Movement Primitives", Nemec et al., 2011
        - [7] "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
        - [8] "A Framework for Learning Biped Locomotion with Dynamical Movement Primitives", Nakanishi et al., 2004
        - [9] "Policy Search for Motor Primitives in Robotics", Kober et al., 2010
        - [10] "A Generalized Path Integral Control Approach to Reinforcement Learning", Theodorou et al., 2010
        - [11] "A correct formulation for the Orientation Dynamic Movement Primitives for robot control in the
          Cartesian space", Koutras et al., 2019
    """

    def __init__(self, canonical_system, forcing_term, y0=0, goal=1, stiffness=None, damping=None):
        """Initialize the DMP.

        Args:
            canonical_system (CS): canonical system which drives the DMP transformation system
            forcing_term (list): list of forcing terms (one forcing term for each DMP). Each forcing term can have
              different number of basis functions.
            y0 (float, np.array[float[M]]): initial state of DMPs
            goal (float, np.array[float[M]]): goal state of DMPs
            stiffness (float): stiffness term in the transformation system for DMPs
            damping (float): damping term in the transformation system for DMPs
        """

        self.cs = canonical_system

        if isinstance(forcing_term, ForcingTerm):
            forcing_term = [forcing_term]
        elif isinstance(forcing_term, (list, tuple)):
            for f in forcing_term:
                if not isinstance(f, ForcingTerm):
                    raise TypeError("An item in the iterable is not an instance of ForcingTerm.")
        else:
            raise TypeError("Expecting forcing term to be an instance of ForcingTerm or a list/tuple of ForcingTerm")

        self.f = forcing_term
        self.num_dmps = len(forcing_term)
        self.dt = self.cs.dt
        self.timesteps = self.cs.timesteps

        # check initial and goal positions # TODO use property to set them
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.num_dmps) * y0
        if isinstance(y0, (list, tuple)):
            y0 = np.array(y0)
        self.y0 = y0
        self.dy0 = np.zeros(self.num_dmps)
        self.ddy0 = np.zeros(self.num_dmps)
        if isinstance(goal, (int, float)):
            goal = np.ones(self.num_dmps) * goal
        elif isinstance(goal, (list, tuple)):
            goal = np.array(goal)
        self.goal = goal
        self._check_offset()

        self.y, self.dy, self.ddy = self.y0, self.dy0, self.ddy0

        # set stiffness and damping coefficient (if not specified, make them critically damped, i.e. D=2\sqrt{K})
        self.D = np.ones(self.num_dmps) * 25. if damping is None else damping
        self.K = self.D**2 / 4. if stiffness is None else stiffness

        # set up the DMP system
        self.prev_s = self.cs.init_phase
        self.reset()

        # target forcing term (keep a copy)
        self.f_target = None

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    ##############
    # Properties #
    ##############

    @property
    def input_size(self):
        """Return the input size of the model."""
        return 1  # 1 canonical system

    @property
    def output_size(self):
        """Return the output size of the model."""
        return len(self.f)

    @property
    def input_shape(self):
        """Return the input shape of the model."""
        return tuple([self.input_size])

    @property
    def output_shape(self):
        """Return the output shape of the model."""
        return tuple([self.output_size])

    @property
    def input_dim(self):
        """Return the input dimension of the model; i.e. len(input_shape)."""
        return len(self.input_shape)

    @property
    def output_dim(self):
        """Return the output dimension of the model; i.e. len(output_shape)."""
        return len(self.output_shape)

    @property
    def num_parameters(self):
        """Return the total number of parameters"""
        return np.array([force.w for force in self.f]).size

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other):
        if not isinstance(other, DMP):
            raise TypeError("Trying to copy an object which is not a DMP")
        if deep:
            return copy.deepcopy(other)
        return copy.copy(other)

    @staticmethod
    def is_parametric():
        """Return True as a DMP has weights that need to be optimized."""
        return True

    @staticmethod
    def is_linear():
        """Return True as a DMP is linear in terms of its weights (i.e. learnable parameters)"""
        return True

    @staticmethod
    def is_recurrent():
        """Return False as a DMP is not a recurrent model."""
        return False

    @staticmethod
    def is_probabilistic():
        """The DMP is a deterministic model."""
        return False

    @staticmethod
    def is_discriminative():
        """The DMP is a discriminative model which predicts the output :math:`y` given the input :math:`x`"""
        return True

    @staticmethod
    def is_generative():
        """The DMP is not a generative model."""
        return False

    ###########
    # Methods #
    ###########

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for force in self.f:
            yield force.w

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        for force in self.f:
            yield str(force), force.w

    def list_parameters(self):
        """Return a list of parameters"""
        return list(self.parameters())

    def hyperparameters(self):
        """Return an iterator over the hyper-parameters."""
        yield self.K
        yield self.D
        # yield basis_functions

    def named_hyperparameters(self):
        """Return an iterator over the hyper-parameters, yielding both the name and the hyper-parameter itself."""
        yield "stiffness", self.K
        yield "damping", self.D

    def list_hyperparameters(self):
        """Return a list of hyper-parameters."""
        return list(self.hyperparameters())

    def get_vectorized_parameters(self, to_numpy=True):
        """Return a vectorized form (1 dimensional array) of the parameters."""
        parameters = self.parameters()
        vector = np.concatenate([parameter.reshape(-1) for parameter in parameters])  # np.concatenate = torch.cat
        # if to_numpy:
        #     return vector.detach().numpy()
        return vector

    def set_vectorized_parameters(self, vector):
        """Set the vector parameters."""
        # convert the vector to torch array
        # if isinstance(vector, np.ndarray):
        #     vector = torch.from_numpy(vector).float()

        # set the parameters from the vectorized one
        # idx = 0
        # for parameter in self.parameters():
        #     size = parameter.nelement()
        #     parameter.data = vector[idx:idx+size].reshape(parameter.shape)
        #     idx += size

        # set the parameters from the vectorized one
        idx = 0
        for force in self.f:
            size = force.w.size
            force.w = vector[idx:idx+size].reshape(force.w.shape)
            idx += size

    def get_damping_ratio(self):
        r"""
        Return the damping ratio :math:`\zeta = D / D_c` where :math:`D_c = 2 \sqrt{K}`.

        * if :math:`\zeta` = 0, the system is undamped (i.e. no damping)
        * if :math:`\zeta` < 1, the system is underdamped (i.e. there will be some oscillations)
        * if :math:`\zeta` = 1, the system is critically damped (i.e. return to equilibrium as fast as possible
          without oscillating).
        * if :math:`\zeta` > 1, the system is overdamped (i.e. the system returns to equilibrium without oscillating
          but might be slow depending on the damping value).
        """
        return self.D / (2*np.sqrt(self.K))

    def _check_offset(self):
        """Check to see if the initial position and goal are the same. If that is the case, offset slightly so that
        the forcing term is not 0. Otherwise, look at the `BioDMP` class.
        """
        self.goal[self.y0 == self.goal] += 1e-4

    def get_scaling_term(self, new_goal=None):
        # this is overridden by the child classes
        return np.ones(self.num_dmps)

    def _generate_goal(self, y_des):
        raise NotImplementedError()

    def reset(self):
        """Reset the transformation and canonical systems"""
        self.y = self.y0.copy()
        self.dy = self.dy0.copy()  # np.zeros(self.num_dmps)
        self.ddy = self.ddy0.copy()
        self.prev_s = self.cs.reset()

    def step(self, s=None, tau=1.0, error=0.0, forcing_term=None, new_goal=None, external_force=None,
             rescale_force=True):
        """Run the DMP transformation system for a single time step.

        Args:
            s (None, float): the phase value. If None, it will use the canonical system.
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            error (float): optional system feedback
            forcing_term (np.array[float[M]]): if given, it will replace the forcing term (where `M` = number of DMPs)
            new_goal (np.array[float[M]]): new goal (where `M` = number of DMPs)
            rescale_force (bool): if the given forcing term should be rescaled.
        """

        # system feedback
        error_coupling = 1.0 / (1.0 + error)

        # get phase from canonical system
        if s is None:
            s = self.cs.step(tau=tau, error_coupling=error_coupling)
        elif not isinstance(s, (float, int)):
            raise TypeError("Expecting the phase 's' to be a float or integer. Instead, I got {}".format(type(s)))

        # check if same phase as before
        if s == self.prev_s:
            return self.y, self.dy, self.ddy

        if new_goal is None:
            new_goal = self.goal

        # save previous position and velocity
        prev_y, prev_dy = self.y.copy(), self.dy.copy()

        # compute scaling factor for the forcing term
        scaling = self.get_scaling_term(new_goal)

        # for each DMP, solve transformation system equation using Euler's method
        for d in range(self.num_dmps):

            # compute forcing term
            if forcing_term is None:
                f = self.f[d](s) * scaling[d]
                # f = self.f_gen(s) * scaling[d]
            else:
                if rescale_force:
                    f = forcing_term[d] * scaling[d]
                else:
                    f = forcing_term[d]

            # DMP acceleration
            self.ddy[d] = self.K[d]/(tau**2) * (new_goal[d] - self.y[d]) - self.D[d]/tau * self.dy[d] + f/(tau**2)
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] / tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * self.dt * error_coupling

        # return self.y, self.dy, self.ddy
        return prev_y, prev_dy, self.ddy

    def rollout(self, timesteps=None, tau=1.0, error=0.0, forcing_term=None, new_goal=None, rescale_force=True,
                **kwargs):
        """Generate position, velocity, and acceleration trajectories, no feedback is incorporated.

        Args:
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            timesteps (None, int): the number of steps to perform
            error (float): optional system feedback
            forcing_term (np.array[float[M,T]]): if given, it will replace the forcing term (shape [num_dmps,
              timesteps])
            new_goal (np.array[float[M]]): new goal (of shape [num_dmps,])

        Returns:
            np.array[float[M,T]]: y (position) trajectories
            np.array[float[M,T]]: dy (velocity) trajectories
            np.array[float[M,T]]: ddy (acceleration) trajectories
        """

        # reset the canonical and transformation systems
        self.reset()

        if timesteps is None:
            timesteps = int(self.timesteps * tau)

        # set up tracking vectors
        y_track = np.zeros((self.num_dmps, timesteps))
        dy_track = np.zeros((self.num_dmps, timesteps))
        ddy_track = np.zeros((self.num_dmps, timesteps))

        # for the other timesteps, solve DMP equation using Euler's method
        for t in range(timesteps):
            if forcing_term is None:
                y, dy, ddy = self.step(tau=tau, error=error, new_goal=new_goal, external_force=None)
            else:
                y, dy, ddy = self.step(tau=tau, error=error, forcing_term=forcing_term[:, t], new_goal=new_goal,
                                       rescale_force=rescale_force)

            # record timestep
            y_track[:, t] = y
            dy_track[:, t] = dy
            ddy_track[:, t] = ddy

        return y_track, dy_track, ddy_track

    def train(self, f_target):
        """Train the forcing terms."""
        # train each forcing term
        if f_target.shape[0] != len(self.f):
            raise ValueError("Mismatch between the number of forcing terms")

        # train each forcing term
        for forcing_term, target in zip(self.f, f_target):
            forcing_term.train(target)

    def imitate(self, y_des, dy_des=None, ddy_des=None, interpolation='cubic', plot=False):
        """Imitate a desired trajectory, and learn the parameters that best realizes it.

        Args:
            y_des (np.array[float[M,T]], np.array[float[N,M,T]]): the desired position trajectories of each DMP with
              shape [num_dmps, timesteps] or [num_trajectories, num_dmps, timesteps]. The number of timesteps for each
              trajectory can be different. Note that each trajectory should have the same initial state and goal. When
              giving multiple trajectories to DMPs, they will be averaged out.
            dy_des (np.array[float[M,T]], np.array[float[N,M,T]]): the desired velocities with shape
              [num_dmps, timesteps] or [num_trajectories, num_dmps, timesteps].
            ddy_des (np.array[float[M,T]], np.array[float[N,M,T]]): the desired accelerations with shape
              [num_dmps, timesteps] or [num_trajectories, num_dmps, timesteps].
            interpolation (str): how to interpolate the data. Select between 'linear', 'cubic', and 'hermite'.
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.goal = self._generate_goal(y_des)
        self._check_offset()

        timesteps = y_des.shape[1]

        def interpolate(x, dt, period, timesteps, new_timesteps, interpolation=interpolation, return_gen=False):
            # generate function to interpolate the desired trajectory
            t = np.linspace(0, period, timesteps)
            if interpolation == 'linear':  # use linear interpolation
                path_gen = scipy.interpolate.interp1d(t, x, axis=-1)
            elif interpolation == 'cubic':  # use cubic spline interpolation
                path_gen = scipy.interpolate.CubicSpline(t, x, axis=-1)
            else:  # TODO: implement hermite (see utils.interpolator.hermite)
                raise ValueError("The requested interpolation has not been implemented. Select between 'linear' or "
                                 "'cubic'")
            if return_gen:
                return path_gen
            return path_gen([t * self.dt for t in range(new_timesteps)])

        y_des = interpolate(y_des, dt=self.dt, period=self.cs.T, timesteps=timesteps,
                            new_timesteps=self.timesteps, interpolation=interpolation)

        # compute desired velocity if necessary
        if dy_des is None:
            # calculate velocity of y_des
            dy_des = np.diff(y_des) / self.dt
            # add zero to the beginning of every row
            dy_des = np.hstack((np.zeros((self.num_dmps, 1)), dy_des))
        else:
            if dy_des.ndim == 1:
                dy_des = dy_des.reshape(1, len(dy_des))
            dy_des = interpolate(dy_des, self.dt, self.cs.T, dy_des.shape[1], self.timesteps,
                                 interpolation=interpolation)
        self.dy0 = dy_des[:, 0].copy()

        # compute desired acceleration if necessary
        if ddy_des is None:
            # calculate acceleration of y_des
            ddy_des = np.diff(dy_des) / self.dt
            # add zero to the beginning of every row
            ddy_des = np.hstack((np.zeros((self.num_dmps, 1)), ddy_des))
        else:
            if ddy_des.ndim == 1:
                ddy_des = ddy_des.reshape(1, len(ddy_des))
            ddy_des = interpolate(ddy_des, self.dt, self.cs.T, ddy_des.shape[1], self.timesteps,
                                  interpolation=interpolation)
        self.ddy0 = ddy_des[:, 0].copy()

        # find the force required to move along this trajectory (with shape [num_dmps, timesteps])
        f_target = ddy_des - self.K.reshape(-1, 1) * (self.goal.reshape(-1, 1) - y_des) + self.D.reshape(-1, 1) * dy_des

        # plot
        if plot:
            plt.figure()
            plt.plot(y_des[0], 'b', label='pos')
            plt.plot(dy_des[0], 'g', label='vel')
            plt.plot(ddy_des[0], 'r', label='acc')
            plt.plot(f_target[0], 'k', label='force')
            plt.legend()
            plt.show()

        # self.f_gen = interpolate(f_target, dt=self.dt, period=self.cs.T, timesteps=timesteps,
        #                          new_timesteps=timesteps, interpolation=interpolation, return_gen=True)

        # efficiently generate weights to realize f_target
        self.f_target = f_target
        self.train(f_target)

        # reset the canonical and transformation systems
        self.reset()

        return y_des

    def get_forcing_term(self, s):
        """
        Get the forcing terms based on the given phase value.

        Args:
            s (float, np.array[float[T]]): phase value(s)

        Returns:
            np.array[float[M]], np.array[float[M,T]]: forcing terms
        """
        return np.array([self.f[d](s) for d in range(self.num_dmps)])

    def generate_goal(self, y0=None, dy0=None, ddy0=None, f0=None):
        """
        Generate the goal from the initial positions, velocities, accelerations, and forces.

        Args:
            y0 (np.array[float[M]], None): initial positions. If None, it will take the default initial positions.
            dy0 (np.array[float[M]], None): initial velocities. If None, it will take the default initial velocities.
            ddy0 (np.array[float[M]], None): initial accelerations. If None, it will take the default initial
              accelerations.
            f0 (np.array[float[M]], None): initial forcing terms. If None, it will compute it based on the learned
              weights. You can also give `dmp.f_target[:,0]` to get the correct goal.

        Returns:
            np.array[float[M]]: goal position for each DMP.
        """
        if y0 is None:
            y0 = self.y0
        if dy0 is None:
            dy0 = self.dy0
        if ddy0 is None:
            ddy0 = self.ddy0
        if f0 is None:
            s0 = self.cs.init_phase
            f0 = self.get_forcing_term(s0)

        return 1/self.K * (ddy0 + self.D * dy0 + self.K * y0 - f0)

    def sequence(self, model, mode=0):
        """
        Define how to sequence with another DMP model.

        Args:
            model (DMP): DMP model
            mode (int): specifies how to sequence the two DMP models.

        Returns:
            DMP: the sequenced model

        References:
            [1] "Action Sequencing using Dynamic Movement Primitives", Nemec et al., 2011
        """
        if not isinstance(model, DMP):
            raise TypeError("The given model is not an instance of DMP.")
        pass

    def plot_rollout(self, ax=None, nrows=1, ncols=1, suptitle=None, titles=None, show=False):
        """
        Plot a complete rollout.

        Args:
            ax (plt.Axes.axis, None): figure axis.
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles (str, list[str]): title for each subplot.
            show (bool): if True, it will show and block the plot.
        """
        # perform a rollout
        y, dy, ddy = self.rollout()

        # if ax is not defined
        if ax is None:
            plt.figure()
        if suptitle is not None:
            plt.suptitle(suptitle)

        # plot each subplot
        for i in range(y.shape[0]):
            plt.subplot(nrows, ncols, i + 1)
            if titles is not None:
                if isinstance(titles, str):
                    plt.title(titles)
                elif isinstance(titles, (list, tuple, np.ndarray)) and i < len(titles):
                    plt.title(titles[i])
            plt.plot(y[i])

        # tight the layout
        plt.tight_layout()
        if show:
            plt.show()

    # def __rshift__(self, other):
    #     """
    #     Sequence DMP model with another learning model.
    #
    #     Ref: "Action Sequencing using Dynamic Movement Primitives", Nemec et al., 2011
    #
    #     :param other: another DMP model
    #     :return:
    #     """
    #     # If we sequence two DMP models
    #     if isinstance(other, DMP):
    #
    #     else:
    #         # if it is another model, call the parent's method which knows how to sequence different models
    #         super(DMP, self).__rshift__(other)

