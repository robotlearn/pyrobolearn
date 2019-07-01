#!/usr/bin/env python
"""Define the Plot class.

Warnings: THIS IS EXPERIMENTAL.

Dependencies:
- `matplotlib`
"""

# TODO

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import multiprocessing


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Plot(object):
    """Plot (abstract) class.

    The plotter allows to plot different things. Notably, it can plot in real-time the joint values, the orientation
    frame of each body.
    """
    pass


class RealTimePlot(Plot):
    """Real-time plotter

    This plotter spawns a new process that is responsible to update a plot in real-time. To achieve that goal, the
    master process sent the data (through the `RealTimePlot.update` method) through a pipe to the new process which
    updates the plot.
    """
    pass
    # def __init__(self, ticks=1, blit=True):
    #     # create pipe, queue, and process
    #     self.pipe, pipe = multiprocessing.Pipe()
    #     self.queue = multiprocessing.Queue()
    #     self.process = multiprocessing.Process(target=self._plot, args=(pipe, self.queue))
    #
    #     # start process
    #     self.process.start()
    #
    # def _plot(self, pipe, queue):
    #     """To be implemented in the child class."""
    #     pass
    #
    # def update(self):
    #     """To be implemented in the child class."""
    #     pass
    #
    # def close(self):
    #     """close the plotter."""
    #     # notify the plot child process
    #     self.pipe.send('END')
    #
    #     # wait for the child process to close
    #     self.process.join()
    #
    #     # close queue and pipe
    #     self.queue.close()
    #     self.pipe.close()
    #
    # def __del__(self):
    #     """Closing the plotter."""
    #     self.close()


class BodyPlot(RealTimePlot):
    r"""Body plotter.

    The Body plotter draws the joint positions with their corresponding frame in a 3D plot.
    """
    pass


class JointPlot(RealTimePlot):
    r"""Joint plotter

    The Joint plotter plots the joint position, velocity, acceleration and torque values.
    """

    def __init__(self, robot, joint_ids=None, position=False, velocity=False, acceleration=False, torque=False,
                 ticks=1, blit=True):
        """
        Initialize the joint plotter.

        Args:
            robot (Robot): robot instance.
            joint_ids (list of int, int, None): joint id(s) to plot.
            position (bool): if True, it will plot the joint positions.
            velocity (bool): if True, it will plot the joint velocities.
            acceleration (bool): if True, it will plot the joint accelerations.
            torque (bool): if True, it will plot the joint torques.
            ticks (int): number of ticks to sleep before sending the new data.
            blit (bool): if we should use blit, that is, if we should re-draw only the parts that have changed.
                If blit = True, it plots faster but can only update what is inside the plot (so not the xticks,
                yticks, xlabel, etc).
        """
        # set variable
        self.robot = robot
        self.joint_ids = joint_ids
        self.plot_position = position
        self.plot_velocity = velocity
        self.plot_acceleration = acceleration
        self.plot_torque = torque
        self.ticks = ticks
        self.cnt = 0
        self.blit = blit

        self.plot_exist = True

        # create pipe, queue, and process
        self.pipe, pipe = multiprocessing.Pipe()
        self.queue = multiprocessing.Queue()
        self.plot_process = multiprocessing.Process(target=self._plot, args=(pipe, self.queue))
        self.plot_process.start()

    def _plot(self, pipe, queue):
        """Plot the streamed data."""

        # set pipe and queue
        self.pipe = pipe
        self.queue = queue

        # create figure, subplots, axes, titles,...
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

        # create line
        line, = ax.plot([], [], lw=2)

        self.x = []
        self.y = []

        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # def gen():
        #     states = self.pipe.recv()
        #     if not (isinstance(states, bool) and states):
        #         yield states
        #     else:
        #         print("Over")

        # animation function.  This is called sequentially
        def animate(i):
            states = self.pipe.recv()
            # if isinstance(states, bool) and states:
            #     self.anim.event_source.stop()

            # print("Received states: {}".format(states))

            self.y.append(states['q'][0])
            self.y = self.y[-10:]

            # print(self.y[:3])

            line.set_data(range(len(self.y)), self.y)
            # ax.set_xlim(0 + 0.01 * i, 2 + 0.01 * i)
            # ax.set_xticklabels(np.linspace(0.01 * i, 2 + 0.01 * i, 5))
            return line,

        # create funcanimation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=None, interval=0.0001, blit=self.blit)
        plt.show()

        # if we get out of the animation, notify the master process
        self.queue.put(True)
        self.pipe.close()
        self.queue.close()

    def update(self):
        """Update the plot by getting the """
        if self.cnt % self.ticks == 0 and self.plot_exist:

            # get useful information
            states = {}
            if self.plot_position:
                states['q'] = self.robot.get_joint_positions(joint_ids=self.joint_ids)
            if self.plot_velocity:
                states['dq'] = self.robot.get_joint_velocities(joint_ids=self.joint_ids)
            if self.plot_acceleration:
                states['ddq'] = self.robot.get_joint_accelerations(joint_ids=self.joint_ids)

            # send the data to the process
            self.pipe.send(states)

        self.cnt += 1

        if not self.queue.empty():
            result = self.queue.get()
            if result:
                print("The animation has finished. Closing process...")
                self.plot_process.join()
                print("Process has been closed.")
            else:
                print("Got result: {}".format(result))


class LinkPlot(RealTimePlot):
    r"""Link plotter

    The Link plotter plots a link position, velocity, acceleration, force along the 3 axis (x,y,z).
    """
    pass


# Tests
if __name__ == '__main__':

    # Try to move the robot in the simulator

    from itertools import count
    import pyrobolearn as prl

    sim = prl.simulators.Bullet()
    world = prl.worlds.BasicWorld(sim)
    robot = world.load_robot('kuka_iiwa')

    plot = JointPlot(robot, joint_ids=[3], position=True, ticks=24)

    for t in count():
        plot.update()
        world.step(sim.dt)
