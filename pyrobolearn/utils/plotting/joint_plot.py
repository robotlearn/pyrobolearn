#!/usr/bin/env python
"""Define the joint real-time plotting class.

Warnings: DON'T FORGET TO CLOSE FIRST THE FIGURE THEN THE SIMULATOR OTHERWISE YOU WILL HAVE THE PLOTTING PROCESS STILL
RUNNING
"""

import numpy as np

from pyrobolearn.utils.plotting.plot import RealTimePlot
import pyrobolearn as prl


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointRealTimePlot(RealTimePlot):
    r"""Joint plotting tool

    The joint plotting tool plots the joint position, velocity, acceleration and torque values in real-time.
    """

    def __init__(self, robot, joint_ids=None, position=False, velocity=False, acceleration=False, torque=False,
                 num_points=100, xlims=None, ylims=None, suptitle='Joint states', ticks=1, blit=True, interval=0.0001):
        """
        Initialize the joint plotting tool.

        Args:
            robot (Robot): robot instance.
            joint_ids (list of int, int, None): joint id(s) to plot. If None, it will take all the actuated joints.
            position (bool): if True, it will plot the joint positions.
            velocity (bool): if True, it will plot the joint velocities.
            acceleration (bool): if True, it will plot the joint accelerations.
            torque (bool): if True, it will plot the joint torques.
            num_points (int): number of points to keep in the plots.
            xlims ((list of) tuple of float, None): x-limits for each subplot.
            ylims ((list of) tuple of float, None): y-limits for each subplot.
            suptitle (str): main title for the subplots.
            ticks (int): number of ticks to sleep before sending the new data.
            blit (bool): if we should use blit, that is, if we should re-draw only the parts that have changed.
                If blit = True, it plots faster but can only update what is inside the plot (so not the xticks,
                yticks, xlabel, etc).
            interval (float): Delay between frames in milliseconds.
        """
        # set robot
        if not isinstance(robot, prl.robots.Robot):
            raise TypeError("Expecting the given 'robot' to be an instance of `Robot`, but got instead: "
                            "{}".format(robot))
        self._robot = robot

        # set joint_ids
        if joint_ids is None:
            joint_ids = self._robot.joints
        if isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self._joint_ids = joint_ids

        nrows, ncols = 1, 1
        if len(joint_ids) <= 4:
            ncols = len(joint_ids)
        else:
            ncols = 4
            if len(joint_ids) % ncols == 0:
                nrows = int(len(joint_ids) / ncols)
            else:
                nrows = int(len(joint_ids) / ncols) + 1

        # set what we should plot
        self._plot_position = bool(position)
        self._plot_velocity = bool(velocity)
        self._plot_acceleration = bool(acceleration)
        self._plot_torque = bool(torque)

        states = np.array([self._plot_position, self._plot_velocity, self._plot_acceleration, self._plot_torque])
        self._num_states = len(states[states])

        if self._num_states == 0:
            raise ValueError("Expecting to plot at least something (position, velocity, acceleration or torque)")

        # set num_points
        self._num_points = num_points if num_points > 10 else 10

        # check xlim and ylim
        if xlims is None:
            xlims = (0, self._num_points)
        if ylims is None:
            ylims = (-2*np.pi, 2*np.pi)

        super(JointRealTimePlot, self).__init__(nrows=nrows, ncols=ncols, xlims=xlims, ylims=ylims,
                                                titles=self._robot.get_joint_names(self._joint_ids),
                                                suptitle=suptitle, ticks=ticks, blit=blit, interval=interval)

    def _init(self, axes):
        """Init the plots by creating the lines in each axis."""
        # create lines
        self._lines = []

        for i, joint_id in enumerate(self._joint_ids):
            if self._plot_position:
                line, = axes[i].plot([], [], lw=self._linewidths[i], color='blue')
                self._lines.append(line)
            if self._plot_velocity:
                line, = axes[i].plot([], [], lw=self._linewidths[i], color='green')
                self._lines.append(line)
            if self._plot_acceleration:
                line, = axes[i].plot([], [], lw=self._linewidths[i], color='red')
                self._lines.append(line)
            if self._plot_torque:
                line, = axes[i].plot([], [], lw=self._linewidths[i], color='purple')
                self._lines.append(line)

        self._x = []
        length = len(self._joint_ids) * self._num_states
        self._ys = [[] for _ in range(length)]

    def _init_anim(self):
        """Init function (plot the background of each frame) that is passed to FuncAnimation. This has to be
        implemented in the child class."""
        for line in self._lines:
            line.set_data([], [])
        return self._lines

    def _set_line(self, joint_idx, line_idx, data, state_name):
        """Set the new data for the line.

        Args:
            joint_idx (int): joint index.
            line_idx (int): line index.
            data (dict): data that was sent through the pipe.
            state_name (str): name of the state; select between {'q', 'dq', 'ddq', 'tau'}
        """
        self._ys[line_idx].append(data[state_name][joint_idx])
        self._ys[line_idx] = self._ys[line_idx][-self._num_points:]
        self._lines[line_idx].set_data(self._x, self._ys[line_idx])
        line_idx += 1
        return line_idx

    def _animate_data(self, i, data):
        """Animate function that is passed to FuncAnimation. This has to be implemented in the child class.

        Args:
            i (int): frame counter.
            data (dict): data that has been received from the pipe.

        Returns:
            tuple of object: list of object to update
        """
        if len(self._x) < self._num_points:
            self._x = range(len(self._x) + 1)

        k = 0
        for j in range(len(self._joint_ids)):
            if self._plot_position:
                k = self._set_line(joint_idx=j, line_idx=k, data=data, state_name='q')
            if self._plot_velocity:
                k = self._set_line(joint_idx=j, line_idx=k, data=data, state_name='dq')
            if self._plot_acceleration:
                k = self._set_line(joint_idx=j, line_idx=k, data=data, state_name='ddq')
            if self._plot_torque:
                k = self._set_line(joint_idx=j, line_idx=k, data=data, state_name='tau')

        # ax.set_xlim(0 + 0.01 * i, 2 + 0.01 * i)
        # ax.set_xticklabels(np.linspace(0.01 * i, 2 + 0.01 * i, 5))
        return self._lines

    def _update(self):
        """This return the next data to be plotted; this has to be implemented in the child class.

        Returns:
            dict: data to be sent through the pipe and that have to be plotted. This will be given to `_animate_data`.
        """
        data = {}
        if self._plot_position:
            data['q'] = self._robot.get_joint_positions(joint_ids=self._joint_ids)
        if self._plot_velocity:
            data['dq'] = self._robot.get_joint_velocities(joint_ids=self._joint_ids)
        if self._plot_acceleration:
            data['ddq'] = self._robot.get_joint_accelerations(joint_ids=self._joint_ids)
        if self._plot_torque:
            data['tau'] = self._robot.get_joint_torques(joint_ids=self._joint_ids)
        return data


# Tests
if __name__ == '__main__':

    # Try to move the robot in the simulator
    # WARNING: DON'T FORGET TO CLOSE FIRST THE FIGURE THEN THE SIMULATOR OTHERWISE YOU WILL HAVE THE PLOTTING PROCESS
    # STILL RUNNING
    from itertools import count

    sim = prl.simulators.Bullet()
    world = prl.worlds.BasicWorld(sim)
    robot = world.load_robot('kuka_iiwa')

    plot = JointRealTimePlot(robot, joint_ids=None, position=True, velocity=False, acceleration=False,
                             torque=False, ticks=24)

    for t in count():
        plot.update()
        world.step(sim.dt)
