# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the link frame real-time plotting class.

Warnings: DON'T FORGET TO CLOSE FIRST THE FIGURE THEN THE SIMULATOR OTHERWISE YOU WILL HAVE THE PLOTTING PROCESS STILL
RUNNING
"""

import pyrobolearn as prl
from pyrobolearn.utils.plotting.plot import RealTimePlot
from pyrobolearn.utils.transformation import get_matrix_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkFrameRealTimePlot(RealTimePlot):
    r"""Link Frame plotting tool

    The link frame plotting tool plots link frames in a 3D plot in real-time.
    """

    def __init__(self, bodies, link_ids=None, xlims=None, ylims=None, zlims=None, suptitle='Link frames',
                 ticks=1, blit=True, interval=0.0001):
        """
        Initialize the link frame plotting tool.

        Args:
            bodies ((list of) Body): body instance(s).
            link_ids ((list of) list of int, (list of) int, None): link frame id(s) to plot. If None, it will take all
                the actuated links. if -1, it will plot the base frames.
            xlims ((list of) tuple of float, None): x-limits for each subplot.
            ylims ((list of) tuple of float, None): y-limits for each subplot.
            zlims ((list of) tuple of float, None): z-limits for each 3d subplot.
            suptitle (str): main title for the subplots.
            ticks (int): number of ticks to sleep before sending the new data.
            blit (bool): if we should use blit, that is, if we should re-draw only the parts that have changed.
                If blit = True, it plots faster but can only update what is inside the plot (so not the xticks,
                yticks, xlabel, etc).
            interval (float): Delay between frames in milliseconds.
        """
        # check bodies
        if not isinstance(bodies, (list, tuple)):
            bodies = [bodies]
        for body in bodies:
            if not isinstance(body, prl.robots.Body):
                raise TypeError("Expecting each body to be an instance of `Body`, but got instead: "
                                "{}".format(type(body)))
        if len(bodies) == 0:
            raise ValueError("Expecting to be given at least one body, but none were provided...")
        self._bodies = bodies

        # get simulator instance
        self._sim = self._bodies[0].simulator

        # check links
        if isinstance(link_ids, int):
            link_ids = [link_ids] * len(self._bodies)
        if link_ids is None:
            link_ids = []
            for body in bodies:
                links = []
                for joint_id in range(body.num_joints):
                    joint_info = self._sim.get_joint_info(body.id, joint_id)
                    if joint_info[2] != self._sim.JOINT_FIXED:
                        links.append(joint_info[0])

                if len(links) == 0:  # if no links found, add the base
                    links.append(-1)

                link_ids.append(links)
        if not isinstance(link_ids, (tuple, list)):
            raise TypeError("Expecting the given link_ids to be a list / tuple of int, but got instead: "
                            "{}".format(type(link_ids)))
        if len(link_ids) != len(self._bodies):
            if len(self._bodies) == 1:  # if one body
                for link_id in link_ids:
                    if not isinstance(link_id, int):
                        raise TypeError("Expecting an int for each link id, but got instead: {}".format(type(link_id)))
                link_ids = [link_ids]
            else:
                raise ValueError("Expecting the number of bodies (={}) to match with the number of set of links "
                                 "(={})".format(len(self._bodies), len(link_ids)))
        self._link_ids = link_ids

        # set limits
        if xlims is None:
            xlims = (-2., 2.)
        if ylims is None:
            ylims = (-2., 2.)
        if zlims is None:
            zlims = (0., 2.)

        # call parent constructor
        super(LinkFrameRealTimePlot, self).__init__(nrows=1, ncols=1, suptitle=suptitle, xlims=xlims, ylims=ylims,
                                                    zlims=zlims, projection='3d', ticks=ticks, blit=blit,
                                                    interval=interval)

    def _init(self, axes):
        """Init the plots by creating the lines for each frame in the main axis."""
        ax = axes[0]

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # create lines;  list of size N_body * N_links/body * 3, where 3 is for (x,y,z)
        self._lines = []

        for link_ids in self._link_ids:
            if isinstance(link_ids, int):
                link_ids = [link_ids]
            for link_id in link_ids:
                # x axis
                line, = ax.plot(xs=[], ys=[], zs=[], lw=self._linewidths[0], color='red')
                self._lines.append(line)

                # y axis
                line, = ax.plot(xs=[], ys=[], zs=[], lw=self._linewidths[0], color='green')
                self._lines.append(line)

                # z axis
                line, = ax.plot(xs=[], ys=[], zs=[], lw=self._linewidths[0], color='blue')
                self._lines.append(line)

    def _init_anim(self):
        """Init function (plot the background of each frame) that is passed to FuncAnimation. This has to be
        implemented in the child class."""
        for line in self._lines:
            line.set_data([], [])
        return self._lines

    def _update_frame(self, line_idx, position, orientation):
        """Update the lines that compose the frames."""
        pos = position
        rot = get_matrix_from_quaternion(orientation)  # (3,3)

        # set data for x, y, z axes  (NOTE: there is no .set_data() for 3 dim data...)
        for i in range(3):
            new_pos = pos + rot[:, i]/10.
            line = self._lines[line_idx+i]
            line.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
            line.set_3d_properties([pos[2], new_pos[2]])

        line_idx += 3
        return line_idx

    def _animate_data(self, i, data):
        """Animate function that is passed to FuncAnimation. This has to be implemented in the child class.

        Args:
            i (int): frame counter.
            data (dict): data that has been received from the pipe.

        Returns:
            tuple of object: list of object to update
        """
        line_idx = 0
        for i, link_ids in enumerate(self._link_ids):
            positions, orientations = data[i]
            for position, orientation in zip(positions, orientations):
                line_idx = self._update_frame(line_idx, position, orientation)

        return self._lines

    def _update(self):
        """This return the next data to be plotted; this has to be implemented in the child class.

        Returns:
            list: data to be sent through the pipe and that have to be plotted. This will be given to `_animate_data`.
        """
        data = []

        for body, link_ids in zip(self._bodies, self._link_ids):
            positions, orientations = self._sim.get_link_frames(body_id=body.id, link_ids=link_ids)
            data.append([positions, orientations])

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
    box = world.load_box([0.7, 0., 0.2], dimensions=(0.2, 0.2, 0.2), color=(0.2, 0.2, 0.8, 1.), return_body=True)

    plot = LinkFrameRealTimePlot([robot, box], link_ids=None, ticks=24)

    for t in count():
        plot.update()
        world.step(sim.dt)
