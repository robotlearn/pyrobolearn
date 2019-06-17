#!/usr/bin/env python
"""Provide the 2D Spring Loaded Inverted Pendulum (SLIP) model.
"""

import numpy as np
import matplotlib.pyplot as plt


__author__ = ["Songyan Xin", "Brian Delhaisse"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Songyan Xin"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SLIP2D(object):
    r"""2D Spring Loaded Inverted Pendulum (SLIP) model

    The axes are :math:`x` and :math:`y`, where the former points to the right and the latter is up.
    The SLIP model consists of a point mass :math:`m`, a massless spring with stiffness :math:`k`, and a rest length
    :math:`l_0`. Three phases (flight, stance, and flight) are involved in one rollout of the running motion separated
    by a touchdown (TD) and takeoff (TO) events. The state of the system is given by the position and velocity of the
    mass :math:`[x, y, \dot{x}, \dot{y}]`, while the angle of the spring :math:`\theta` is the control parameter.
    """

    def __init__(self, mass, length, stiffness, position=(0., 1.), velocity=(0., 0.), angle=0.,
                 angle_limits=(0, 2*np.pi), gravity=9.81, dt=1e-3):
        """
        Initialize the 2D SLIP model.

        Args:
            mass (float): mass.
            length (float): the rest length of the spring.
            stiffness (float): spring stiffness.
            position (np.float[2]): initial position.
            velocity (np.float[2]): initial velocity.
            angle (float): initial angle
            angle_limits (tuple of 2 floats): angle limits (lower bound, upper bound).
            gravity (float): gravity in the z direction.
            dt (float): integration time step.
        """
        # properties
        self.m = mass
        self.l0 = length
        self.k = stiffness
        self.g = gravity
        self.dt = dt

        # state
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        self.acc = np.zeros(self.pos.shape)

        # control
        self.theta = angle

    ##############
    # Properties #
    ##############

    @property
    def state(self):
        """
        Return the state of the system, i.e. the position and velocity vectors.
        """
        return np.concatenate((self.pos, self.vel))

    ###########
    # Methods #
    ###########

    def kinetic_energy(self, vel=None):
        """
        Return the kinetic energy of the inverted pendulum.

        Args:
            vel (np.ndarray, None): velocity of the inverted pendulum

        Returns:
            float: kinetic energy
        """
        if vel is None:
            vel = self.vel
        return 0.5 * self.m * vel.dot(vel)

    def potential_energy(self, pos=None):
        """
        Return the potential energy due to gravity.

        Args:
            pos (np.ndarray, None): position of the inverted pendulum

        Returns:
            float: potential energy
        """
        if pos is None:
            pos = self.pos
        return self.m * self.g * pos[-1]

    def energy(self, X=None):
        """
        Compute the total energy :math:`E=K+P` where :math:`K` is the kinetic energy of the system, and :math:`P`
        is the potential energy.

        Args:
            X (np.array, None): state [pos, vel] of the inverted pendulum

        Returns:
            float: total energy
        """
        if X is None:
            X = self.state
        P = self.potential_energy(X[:len(X)/2])
        K = self.kinetic_energy(X[len(X)/2:])
        E = K + P
        return E

    def max_velocity(self, Emax):
        """
        Compute the max velocity in the x-direction when in the flight phase, given the total energy of the system
        and assuming that this energy is conserved.

        Args:
            Emax (float): maximum total energy of the system

        Returns:
            float: velocity in the x direction
        """
        P_min = self.m * self.g * self.l0
        K_max = Emax - P_min    # conservation of energy
        v_max = np.sqrt(2 * K_max / self.m)
        return v_max

    def apex_energy(self, y, vx):
        """
        Compute the energy at the apex (which is the maximum height and is the point where the velocity in the
        y-direction is equal to 0).

        Args:
            y (float): height
            vx (float): velocity in the x-direction

        Returns:
            float: total energy
        """
        P = self.m * self.g * y
        K = 0.5 * self.m * (vx ** 2)
        E = K + P
        return E

    def apex_height_to_vel(self, E, y):
        """
        Return the velocity in the x-direction at the apex given the energy.

        Args:
            E (float): total energy of the system
            y (float): apex height of the pendulum

        Returns:
            float: velocity in x
        """
        P = self.m * self.g * y
        K = E - P
        v = np.sqrt(2 / self.m * K)
        return v

    def apex_vel_to_height(self, E, vx):
        """
        Return the apex height given the velocity in the x-direction and the energy of the system.

        Args:
            E (float): total energy of the system
            vx (float): velocity in the x-direction at the apex point

        Returns:
            float: apex height of the system
        """
        K = 0.5 * self.m * vx**2
        P = E - K
        y = P / (self.m * self.g)
        return y

    def apex_height_to_state(self, E, y, x=0):
        """
        Return the state at the given apex height and energy level.

        Args:
            E (float): total energy of the pendulum
            y (float): height of the apex point

        Returns:
            np.array: state of the inverted pendulum
        """
        vx = self.apex_height_to_vel(E, y)
        X = np.array([x, y, vx, 0])
        return X

    def apex_vel_to_state(self, E, vx, x=0):
        """
        Return the state at the given apex velocity in the x-direction and energy level.

        Args:
            E (float): total energy of the system
            vx (float): velocity in the x-direction at the apex point

        Returns:
            np.array: state of the system
        """
        y = self.apex_vel_to_height(E, vx)
        X = np.array([x, y, vx, 0])
        return X

    def in_stance_phase(self, pos, foot_pos):
        """
        Return True if in stance phase.

        Args:
            pos (np.array): position of the inverted pendulum
            foot_pos (np.array): foot position

        Returns:
            bool: True if in stance phase
        """
        return np.linalg.norm(pos - foot_pos) <= self.l0 + 1e-9

    def in_flight_phase(self, pos, foot_pos):
        """
        Return True if in flight phase.

        Args:
            pos (np.array): position of the inverted pendulum
            foot_pos (np.array): foot position

        Returns:
            bool: True if in flight phase
        """
        return not self.in_stance_phase(pos, foot_pos)

    # Dynamics
    def flight_dynamic(self, X=None):
        r"""
        Compute the dynamics of the system during the flight phase; the system is not in contact with the ground.
        In this phase, the mass follows a ballistic projectile trajectory formulated by:

        .. math::

            \ddot{x} &= 0 \\
            \ddot{y} &= -\frac{g}{m}

        Args:
            X (np.array, None): state of the inverted pendulum

        Returns:
            np.array: derivative of the state
        """
        if X is None:
            X = self.state
        x, y, dx, dy = X
        ddx, ddy = 0.0, - self.g / self.m
        self.acc = np.array([ddx, ddy])
        dX = np.array([dx, dy, ddx, ddy])
        return dX

    def stance_dynamic(self, X=None, foot_pos=(0.,0.)):
        r"""
        Compute the dynamics of the system during the stance phase; the system is in contact with the ground.

        .. math::

            \ddot{x} &= k (x - x_f) (l0 - l) / (m l) \\
            \ddot{y} &= k (y - y_f) (l0 - l) / (m l)

        Args:
            X (np.array, None): state of the inverted pendulum
            foot_pos (np.array): foot position

        Returns:

        """
        if X is None:
            X = self.state
        x, y, dx, dy = X
        xf, yf = foot_pos
        l = np.sqrt((x - xf) ** 2 + (y - yf) ** 2)
        ddx = self.k * (x - xf) * (self.l0 - l) / (self.m * l)
        ddy = self.k * (y - yf) * (self.l0 - l) / (self.m * l) - self.g
        self.acc = np.array([ddx, ddy])
        dX = np.array([dx, dy, ddx, ddy])
        return dX

    def step(self):
        """
        Perform one step.

        Returns:
            float [4]: next state
        """
        pass
        # # check which phase we are in
        # if self.in_stance_phase(X[:2], ):
        #     pass
        # else: # flight phase
        #     pass

    def rollout(self, E, vx, theta0=None, plot=False):
        """
        Perform a complete rollout which has three phases (flight, stance, and flight) separated by a touchdown (TD)
        and takeoff (TO) events. It starts from the initial apex state and finish when reaching the other apex
        state.

        Args:
            E (float): total energy of the system
            vx (float): initial velocity in the x direction
            theta0 (float, None): initial angle
            plot (bool): if True, plot.

        Returns:
            float[4]: initial apex state
            float[4]: touchdown state
            float[4]: takeoff state
            float[4]: final apex state
            float[3*T]: time trajectory
            float[4,3*T]: state trajectory
        """
        if theta0 is None:
            theta0 = self.theta

        # compute initial apex state
        X = self.apex_vel_to_state(E, vx)
        X_init, t = np.copy(X), 0.

        # free fall and compute touchdown position
        T_fall, X_TD, t_fall, X_fall = self.free_fall(X, theta0)
        t += T_fall
        foot_pos_TD = np.array([X_TD[0] + self.l0 * np.sin(theta0), 0])

        # stance phase and compute takeoff position
        X = np.copy(X_TD)
        X_stance, t_stance = [], []
        xf_TD, yf_TD = foot_pos_TD
        while np.sqrt((X[0] - xf_TD) ** 2 + (X[1] - yf_TD) ** 2) <= self.l0 + 1e-9: #self.in_stance_phase(X_TD[:2], foot_pos_TD):
            X += self.dt * self.stance_dynamic(X, foot_pos=foot_pos_TD)
            t += self.dt
            X_stance.append(X)
            t_stance.append(t)
        X_TO = np.copy(X)
        X_stance, t_stance = np.array(X_stance).T, np.array(t_stance).T

        # flight phase and compute final apex position
        T_rise, X_apex, t_rise, X_rise = self.free_rise(X)
        t_rise += t
        t += T_rise
        X_final = X_apex

        # combine free-fall, stance, and flight phases
        ts = np.concatenate((t_fall, t_stance, t_rise))
        Xs = np.hstack((X_fall, X_stance, X_rise))

        if plot is True:
            if X_TO[3] < 0: # touchdown vx < 0
                plt.plot(Xs[0], Xs[1], 'k--')
            else:
                plt.plot(Xs[0], Xs[1])
            plt.axis('equal')
            plt.pause(0.001)

        return X_init, X_TD, X_TO, X_final, ts, Xs

    def free_fall(self, X_apex, theta, dt=1e-3):
        """
        Free fall phase.

        Args:
            X_apex (np.array): apex state

        Returns:
            float: rising time
            float[4]: touchdown state (pos and vel)
            float[T]: time trajectory
            float[4,T]: state trajectory
        """
        x, y, dx, dy = X_apex

        # get total distance and time to fall
        y_fall = y - self.l0 * np.cos(theta)
        T_fall = np.sqrt(2 * y_fall / self.g)

        # compute touchdown state
        x_TD, y_TD = x + dx * T_fall, y - y_fall
        dx_TD, dy_TD = dx, dy - self.g * T_fall
        X_TD = np.array([x_TD, y_TD, dx_TD, dy_TD])

        # generate the whole trajectory
        T = np.arange(start=0, stop=T_fall, step=dt)
        X, Y = x + dx * T, y + dy * T - 0.5 * self.g * T ** 2
        dX, dY = dx * np.ones(T.shape), dy - self.g * T
        X_traj = np.vstack((X, Y, dX, dY))

        return T_fall, X_TD, T, X_traj

    def free_rise(self, X_TO, dt=1e-3):
        """
        Free rise flight phase.

        Args:
            X_TO (np.array): take off state

        Returns:
            float: rising time
            float[4]: apex state (pos and vel)
            float[T]: time trajectory
            float[4,T]: state trajectory
        """
        x, y, dx, dy = X_TO

        # get total time and distance to rise
        T_rise = dy / self.g
        y_rise = 0.5 * self.g * T_rise ** 2

        # compute apex state
        x_apex, y_apex = x + dx * T_rise, y + y_rise
        dx_apex, dy_apex = dx, 0.0
        X_apex = np.array([x_apex, y_apex, dx_apex, dy_apex])

        # generate the whole trajectory
        T = np.arange(start=0, stop=T_rise, step=dt)
        X, Y = x + dx * T, y + dy * T - 0.5 * self.g * T**2
        dX, dY = dx * np.ones(T.shape), dy - self.g * T
        X_traj = np.vstack((X, Y, dX, dY))

        return T_rise, X_apex, T, X_traj

    def stance_GRF(self, X=None, foot_pos=(0,0)):
        """
        Compute the ground reaction forces during the stance phase.

        Args:
            X (np.array, None): state of the inverted pendulum.
            foot_pos (np.array): foot position

        Returns:
            np.array: ground reaction forces
        """
        if X is None:
            X = self.state
        x, y, dx, dy = X
        xf, yf = foot_pos
        l = np.sqrt((x - xf) ** 2 + (y - yf) ** 2)
        leg_force = self.k * (self.l0 - l)
        leg_angle = np.arctan2(y - yf, x - xf)

        # print("leg_angle: ", np.rad2deg(leg_angle))
        GRF_x = leg_force * np.cos(leg_angle)
        GRF_y = leg_force * np.sin(leg_angle)
        return np.array([GRF_x, GRF_y])

    def plot_trajectory(self, X, title='', block=True):
        """
        Plot state trajectory.
        """
        fig = plt.figure()
        plt.plot(X[0], X[1])
        plt.plot([X[0, 0]], [X[1, 0]], marker='o', markersize=3, color='green')
        plt.plot([X[0, -1]], [X[1, -1]], marker='o', markersize=3, color='red')
        plt.xlim(np.min(X[0]) - 0.1, np.max(X[0]) + 0.2)
        plt.ylim(0, np.max(X[1]) + 0.2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.show(block=block)


# Tests
if __name__ == '__main__':

    # define some variables
    mass = 85.0

    # create 2D slip model
    theta_min, theta_max = 0., np.deg2rad(15)
    theta = np.random.uniform(theta_min, theta_max)
    model = SLIP2D(mass=mass, length=0.8, stiffness=mass*500, gravity=9.81, dt=1.e-3, angle=theta)

    # plot trajectory
    E = 750
    vx = np.random.uniform(0, model.max_velocity(E))
    X = model.rollout(E=E, vx=vx)[-1]
    model.plot_trajectory(X, title='E: {} - theta: {:.2} - vx: {:.2}'.format(str(E), str(np.rad2deg(theta)), str(vx)))