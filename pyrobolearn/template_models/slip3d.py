#!/usr/bin/env python
"""Provide the 3D Spring Loaded Inverted Pendulum (SLIP) model.
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


class SLIP3D(object):
    r"""3D Spring Loaded Inverted Pendulum (SLIP) model

    The axes are :math:`(x,y,z)`, where :math:`x` points in front, :math:`y` to the left, and :math:`z` is up.
    The SLIP model consists of a point mass :math:`m`, a massless spring with stiffness :math:`k`, and a rest length
    :math:`l_0`. Three phases (flight, stance, and flight) are involved in one rollout of the running motion separated
    by a touchdown (TD) and takeoff (TO) events. The state of the system is the position and velocity of the mass
    :math:`[x, y, z, \dot{x}, \dot{y}, \dot{z}]`, while the 2 angles :math:`[\theta, \phi]` are the control
    parameters.

    References:
        [1] "A Dual-SLIP Model For Dynamic Walking In A Humanoid Over Uneven Terrain" (Diss), Yiping, 2015.
    """

    def __init__(self, mass, length, stiffness, position=(0., 0., 1.), velocity=(0., 0., 0.), angles=(0., 0.),
                 angle_limits=(0, 2*np.pi), gravity=9.81, dt=1e-3):
        """
        Initialize the 3D SLIP model.

        Args:
            mass (float): mass.
            length (float): the rest length of the spring.
            stiffness (float): spring stiffness.
            position (np.float[3]): initial position.
            velocity (np.float[3]): initial velocity.
            angle (np.float[2]): initial angles (pitch, roll).
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
        self.angles = np.array(angles)

    ##############
    # Properties #
    ##############

    @property
    def state(self):
        """
        Return the state of the system, i.e. the position and velocity vectors.
        """
        return np.concatenate((self.pos, self.vel))

    @property
    def control(self):
        """
        Return the control input of the system, i.e. the two angles.
        """
        return self.angles

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

    def check_energy(self, key_state):
        X0, X_touchdown, X_takeoff, X_apex = key_state
        for X in key_state:
            print("X: ", X)
            print("E: ", self.energy(X))

    def foot_position(self, X=None, U=None):
        """
        Compute the foot position given the state and control law.

        Args:
            X (np.array): state of the SLIP model
            U (np.array): control law

        Returns:
            np.array: foot position
        """
        if X is None: X = self.state
        if U is None: U = self.control
        x, y, z, dx, dy, dz = X
        theta, phi = U
        return np.array([x, y, z]) + self.l0 * np.array([np.sin(theta) * np.cos(phi),
                                                         np.sin(theta) * np.sin(phi),
                                                         -np.cos(theta)])

    # Dynamics
    def flight_dynamics(self, X=None):
        """
        Compute the flight dynamics of the SLIP model given the state.

        Args:
            X (np.array, None): state of the system

        Returns:
            np.array: derivative of the state
        """
        if X is None: X = self.state
        x, y, z, dx, dy, dz = X
        return np.array([dx, dy, dz, 0, 0, self.g])

    def stance_dynamics(self, X, foot_pos):
        """
        Compute the stance dynamics of the SLIP model given the state and foot position.

        Args:
            X (np.array): state of the system
            foot_pos (np.array): foot position

        Returns:
            np.array: derivative of the state
        """
        x, y, z, dx, dy, dz = X
        l = np.array([x, y, z]) - foot_pos
        l_unit = l / np.linalg.norm(l)
        (ddx, ddy, ddz) = self.k / self.m * (self.l0 - np.linalg.norm(l)) * l_unit + np.array([0, 0, self.g])
        return np.array([dx, dy, dz, ddx, ddy, ddz])

    # Phases
    def flight_fall_phase(self, X, U):
        x, y, z, dx, dy, dz = X
        theta, phi = U
        cur_foot_pos = self.foot_position(X, U)
        # print("cur foot pos: ", cur_foot_pos)
        if cur_foot_pos[2] <= 0.0:
            print("foot penetrate into the ground!")
        else:
            fall_duration = np.sqrt(2 * cur_foot_pos[2] / np.abs(g))
            # tt = np.arange(0, fall_duration, dt)
            tt = np.linspace(0, fall_duration, num=fall_duration / self.dt)
            xx = x + dx * tt
            yy = y + dy * tt
            zz = z + dz * tt + 0.5 * self.g * tt ** 2
            dxx = dx * np.ones(tt.shape)
            dyy = dy * np.ones(tt.shape)
            dzz = dz + self.g * tt
            fall_traj = np.vstack((xx, yy, zz, dxx, dyy, dzz)).T
            X_touchdown = (xx[-1], yy[-1], zz[-1], dxx[-1], dyy[-1], dzz[-1])
            touchdown_foot_pos = self.foot_position(X_touchdown, U)
            return fall_traj, X_touchdown, touchdown_foot_pos

    def stance_phase(self, X, foot_pos):
        XX = []
        while np.linalg.norm(X[:3] - foot_pos) <= self.l0 + 1e-9:
            X = X + self.dt * self.stance_dynamics(X, foot_pos)
            XX.append(X)
        X_takeoff = X
        return np.asarray(XX), X_takeoff, foot_pos

    def flight_rise_phase(self, X):
        x, y, z, dx, dy, dz = X

        if zd <= 0.0:
            print("Take off speed nagetive!")
            return None, None
        else:
            rise_duration = dz / np.abs(g)
            # tt = np.arange(0, rise_duration, dt)
            tt = np.linspace(0, rise_duration, num=rise_duration / self.dt)
            xx = x + dx * tt
            yy = y + dy * tt
            zz = z + dz * tt + 0.5 * self.g * tt ** 2
            dxx = dx * np.ones(tt.shape)
            dyy = dy * np.ones(tt.shape)
            dzz = dz + self.g * tt
            rise_traj = np.vstack((xx, yy, zz, dxx, dyy, dzz)).T
            X_apex = (xx[-1], yy[-1], zz[-1], dxx[-1], dyy[-1], dzz[-1])
            return rise_traj, X_apex

    def step(self):
        pass

    def rollout(self):
        X0, U0 = self.X0, self.U0
        traj_freefall, X_touchdown, pos_touchdown = self.flight_fall_phase(X0, U0)
        traj_stance, X_takeoff, pos_takeoff = self.stance_phase(X_touchdown, pos_touchdown)
        traj_freerise, X_apex = self.flight_rise_phase(X_takeoff)
        return (traj_freefall, traj_stance, traj_freerise), (X0, X_touchdown, X_takeoff, X_apex), pos_touchdown

    # alias
    cycle = rollout

    def plot_cycle(self, traj, key_state, pos_touchdown, ax):
        (traj_freefall, traj_stance, traj_freerise) = traj
        (X0, X_touchdown, X_takeoff, X_apex) = key_state

        # fig = plt.figure(111)
        # ax = fig.add_subplot(111, projection='3d')
        worldFrame = SimpleFrame(ax, pos=[0, 0, 0], quat=trans.quaternion_from_euler(0, 0, 0, axes='sxyz'))

        ax.plot(traj_freefall[:, 0], traj_freefall[:, 1], traj_freefall[:, 2], 'r', label='free fall trajec')
        ax.plot(traj_stance[:, 0], traj_stance[:, 1], traj_stance[:, 2], 'g', label='stance trajec')
        ax.plot(traj_freerise[:, 0], traj_freerise[:, 1], traj_freerise[:, 2], 'b', label='free rise trajec')
        ax.scatter(pos_touchdown[0], pos_touchdown[1], pos_touchdown[2], c="g", s=30)

        # plot leg

        ax.plot([traj_stance[0, 0], pos_touchdown[0]],
                [traj_stance[0, 1], pos_touchdown[1]],
                [traj_stance[0, 2], pos_touchdown[2]],
                color='g', linestyle=':')

        ax.plot([traj_stance[-1, 0], pos_touchdown[0]],
                [traj_stance[-1, 1], pos_touchdown[1]],
                [traj_stance[-1, 2], pos_touchdown[2]],
                color='g', linestyle=':')

        # ax.legend()
        # plt.show()
        plt.draw()
        plt.pause(0.01)

    def plot(self, x, y, z, c, xlabel, ylabel, zlabel, title=None):
        # plot 3D surface with color

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))
        if title is not None:
            plt.title(title)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        scatter = ax.scatter(x, y, z, c=c[:, 0], cmap='coolwarm')  # plt.hot())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        bar_ax = fig.colorbar(scatter, shrink=0.5, aspect=5)
        bar_ax.set_label('theta')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        scatter = ax.scatter(x, y, z, c=c[:, 1], cmap='coolwarm')  # plt.hot())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        bar_ax = fig.colorbar(scatter, shrink=0.5, aspect=5)
        bar_ax.set_label('phi')


# Tests
if __name__ == '__main__':
    pass