#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Plot class.

Warnings: THIS IS EXPERIMENTAL.

Dependencies:
- `matplotlib`
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    The plotting tool allows to plot different things. Notably, it can plot in real-time the joint values, the
    orientation frame of each body.
    """

    def __init__(self, nrows=1, ncols=1, suptitle=None, titles=None, xlims=None, ylims=None, zlims=None, linewidths=1,
                 colors=None, legend=True, projection='2d'):
        """
        Initialize the plot.

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles ((list of) str): title for each subplot.
            xlims ((list of) tuple of float, None): x-limits for each subplot.
            ylims ((list of) tuple of float, None): y-limits for each subplot.
            zlims ((list of) tuple of float, None): z-limits for each 3d subplot.
            linewidths ((list of) int): linewidth for each subplot.
            colors ((list of) str, (list of) tuple of float, None): colors described as strings, or tuple of floats
                (for each channel in RGB or RGBA) for each subplot.
            legend (bool): if True, it will add a legend.
            projection (str): projection, select between {'2d', '3d'}. If '3d', it will plot in 3D.
        """
        self._nrows = nrows
        self._ncols = ncols
        self._nplots = nrows * ncols
        self._suptitle = suptitle

        def to_list(vars, name):
            if not isinstance(vars, list):
                vars = [vars] * self._nplots
            # if len(vars) != self._nplots:
            #     raise ValueError("Expecting the given '" + name + "' to be a list of the same length of the number "
            #                      "subplots")
            return vars

        self._projection = projection
        self._titles = to_list(titles, 'titles')
        self._xlims = to_list(xlims, 'xlims')
        self._ylims = to_list(ylims, 'ylims')
        if self._projection == '3d':
            self._zlims = to_list(zlims, 'zlims')
        else:
            self._zlims = None
        self._linewidths = to_list(linewidths, 'linewidths')
        self._colors = to_list(colors, 'colors')
        self._legend = legend


class RealTimePlot(Plot):
    """Real-time plot tool

    This plot class spawns a new process that is responsible to update a plot in real-time. To achieve that goal, the
    master process sent the data (through the `RealTimePlot.update` method) through a pipe to the new process which
    updates the plot.
    """

    def __init__(self, nrows=1, ncols=1, suptitle=None, titles=None, xlims=None, ylims=None, zlims=None,
                 linewidths=None, colors=None, legend=True, projection='2d', ticks=1, blit=True, interval=0.0001):
        """
        Initialize the real-time plotting tool.

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            suptitle (str): main title for the subplots.
            titles ((list of) str): title for each subplot.
            xlims ((list of) tuple of float, None): x-limits for each subplot.
            ylims ((list of) tuple of float, None): y-limits for each subplot.
            zlims ((list of) tuple of float, None): z-limits for each 3d subplot.
            linewidths ((list of)): linewidth for each subplot.
            colors ((list of) str, (list of) tuple of float, None): colors described as strings, or tuple of floats
                (for each channel in RGB or RGBA) for each subplot.
            legend (bool): if True, it will add a legend.
            projection (str): projection, select between {'2d', '3d'}. If '3d', it will plot in 3D.
            ticks (int): number of ticks to sleep before sending the new data.
            blit (bool): if we should use blit, that is, if we should re-draw only the parts that have changed.
                If blit = True, it plots faster but can only update what is inside the plot (so not the xticks,
                yticks, xlabel, etc).
            interval (float): Delay between frames in milliseconds.
        """
        # init parent class
        super(RealTimePlot, self).__init__(nrows=nrows, ncols=ncols, suptitle=suptitle, titles=titles, xlims=xlims,
                                           ylims=ylims, zlims=zlims, linewidths=linewidths, colors=colors,
                                           legend=legend, projection=projection)

        # set variables for real plot
        self._ticks = ticks
        self._cnt = 0
        self._blit = blit
        self._interval = interval

        self._plot_exist = True

        # create pipe, queue, and process
        self._pipe, pipe = multiprocessing.Pipe()
        self._queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(target=self._plot_process, args=(pipe, self._queue))

        # start process
        self._process.start()

    def _plot_process(self, pipe, queue):
        """Initialize the plot in the child process."""
        # set pipe and queue
        self._pipe = pipe
        self._queue = queue

        # create subplots
        if self._projection == '3d':
            fig = plt.figure()  # figsize=plt.figaspect(0.5))
            axes = []
            for i in range(self._nplots):
                axes.append(fig.add_subplot(self._nrows, self._ncols, i+1, projection='3d'))
        else:
            fig, axes = plt.subplots(nrows=self._nrows, ncols=self._ncols)
        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)
        axes = axes.reshape(-1)
        self._fig, self._axes = fig, axes

        # create main figure title
        if self._suptitle is not None:
            fig.suptitle(self._suptitle)

        # set titles, xlims, and ylims
        for ax, xlim, ylim, title in zip(axes, self._xlims, self._ylims, self._titles):
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if title is not None:
                ax.set_title(title)

        if self._zlims is not None:
            for zlim in self._zlims:
                ax.set_zlim(zlim)

        # tight layout
        fig.tight_layout()

        # def gen():
        #     states = self.pipe.recv()
        #     if not (isinstance(states, bool) and states):
        #         yield states
        #     else:
        #         print("Over")

        # initialize variables
        self._init(axes)

        # create funcanimation
        anim = animation.FuncAnimation(fig, self._animate, init_func=self._init_anim,
                                       frames=None, interval=self._interval, blit=self._blit)
        plt.show()

        # if we get out of the animation, notify the master process
        self._queue.put(True)
        self._pipe.close()
        self._queue.close()

    def _init(self, axes):
        """Initialization of other variables before creating the animation. You can for instance create the various
        lines here."""
        pass

    def _init_anim(self):
        """Init function (plot the background of each frame) that is passed to FuncAnimation. This has to be
        implemented in the child class."""
        raise NotImplementedError

    def _animate(self, i):
        """Animate function that is passed to FuncAnimation.

        Args:
            i (int): frame counter.

        Returns:
            tuple of object: list of object to update
        """
        # receive data from the pipe
        data = self._pipe.recv()

        # animate the data
        return self._animate_data(i, data)

    def _animate_data(self, i, data):
        """Animate function that is passed to FuncAnimation. This has to be implemented in the child class.

        Args:
            i (int): frame counter.
            data (dict): data that has been received from the pipe.

        Returns:
            tuple of object: list of object to update
        """
        raise NotImplementedError

    def update(self):
        """Update the plot: this call `_update` and send the resulting data through the pipe."""
        # if time to update
        if self._cnt % self._ticks == 0 and self._plot_exist:

            # get data and send it to the process through the pipe
            data = self._update()
            self._pipe.send(data)

        self._cnt += 1

        if not self._queue.empty():
            result = self._queue.get()
            if result:
                print("The animation has finished. Closing process...")
                self._process.join()
                print("Process has been closed.")
            else:
                print("Got result: {}".format(result))

    def _update(self):
        """This return the next data to be plotted; this has to be implemented in the child class.

        Returns:
            dict: data to be sent through the pipe and that have to be plotted. This will be given to `_animate_data`.
        """
        raise NotImplementedError

    def close(self):
        """close the plotting tool."""
        # notify the plot child process
        self._pipe.send('END')

        # wait for the child process to close
        self._process.join()

        # close queue and pipe
        self._queue.close()
        self._pipe.close()

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __del__(self):
        """Closing the plotting tool."""
        self.close()

    def __call__(self):
        """update the plot."""
        self.update()


# class LinkPlot(RealTimePlot):
#     r"""Link plotting tool
#
#     The Link plotting tool plots a link position, velocity, acceleration, force along the 3 axis (x,y,z).
#     """
#     pass
