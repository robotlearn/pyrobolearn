#!/usr/bin/env python
"""Defines the various metrics used in different learning paradigms.
"""

import collections
import numpy as np
import matplotlib.pyplot as plt


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Metric(object):
    r"""Metric (abstract) class

    The metric class contains the various metrics used to evaluate a certain learning paradigm (e.g. imitation
    learning, reinforcement learning, transfer learning, active learning, and so on).

    It notably contains the functionalities to evaluate a certain task using the metric, and to plot them.
    """

    def __init__(self, metrics=None):
        """
        Initialize the metric object.

        Args:
            metrics (None, Metric, list of Metric): inner metric objects. Each metric will be plot in a subplot.
        """
        self._metrics = metrics

    ##############
    # Properties #
    ##############

    @property
    def metrics(self):
        """Return the inner list of metric objects."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Set the inner list of metrics."""
        if metrics is None:
            metrics = []
        if not isinstance(metrics, collections.Iterable):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            if not isinstance(metric, Metric):
                raise TypeError("Expecting the given {}th metric to be an instance of `Metric`, instead got: "
                                "{}".format(i, type(metric)))
        self._metrics = metrics

    ###########
    # Methods #
    ###########

    def append(self, metric):
        """Append the given metric to the list of metrics."""
        if not isinstance(metric, Metric):
            raise TypeError("Expecting the given metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(metric)))
        self.metrics.append(metric)

    def update(self, *args, **kwargs):
        """Update the metrics."""
        pass

    def step_update(self, step_idx=None):
        """Update at each time step."""
        if self.metrics:
            for metric in self.metrics:
                metric._step_update(step_idx=step_idx)
        else:
            self._step_update(step_idx=step_idx)

    def _step_update(self, step_idx=None):
        """Update at each time step; this has to be implemented in the child class."""
        pass

    def episode_update(self, episode_idx=None):
        """Update at each episode."""
        if self.metrics:
            for metric in self.metrics:
                metric._episode_step(episode_idx=episode_idx)
        else:
            self._episode_update(episode_idx=episode_idx)

    def _episode_update(self, episode_idx=None):
        """Update at each episode; this has to be implemented in the child class."""
        pass

    def _plot(self, ax):
        """
        Plot the metric in the given axis. This has to be implemented in the child classes.
        """
        pass

    def plot(self, nrows=-1, ncols=-1, block=False, filename=None):
        """
        Plot the metric(s).

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            ax (plt.Axes): axis to plot the figure.
            block (bool): if True, it will block when showing the graphs.
            filename (str, None): if a string is given, it will save the plot in the given filename.

        Returns:
            matplotlib.figure.Figure: figure
            np.array of matplotlib.axes._subplots.AxesSubplot: axes
        """
        metrics = self.metrics if self.metrics else [self]

        # get nrows and ncols
        if nrows < 1 or ncols < 1:
            if nrows < 1 and ncols < 1:
                if len(metrics) <= 4:
                    if len(metrics) <= 2:
                        nrows = 1
                        ncols = len(metrics)
                    else:
                        nrows = 2
                        ncols = int(len(metrics) / 2)
                else:
                    ncols = 4

            if nrows < 1:  # ncols is given
                nrows = int(len(metrics) / ncols)
                if len(metrics) % ncols != 0:
                    nrows += 1

            elif ncols < 1:  # nrows is given
                ncols = int(len(metrics) / nrows)
                if len(metrics) % nrows != 0:
                    ncols += 1

        # get number of subplots
        nplots = nrows * ncols

        # create figure and axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)
        axes = axes.reshape(-1)

        # plot each metric
        for i, metric in enumerate(self.metrics):
            metric._plot(ax=axes[i])

        # save figure if specified
        if filename is not None:
            fig.savefig(filename)

        # show plot
        plt.show(block=block)

        # return figure and axes
        return fig, axes

    #############
    # Operators #
    #############

    # def __repr__(self):
    #     """Return a representation string of the object."""
    #     if self.metrics:
    #         return ' + '.join(self.metrics)
    #     return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        if self.metrics:
            return ' + '.join(self.metrics)
        return self.__class__.__name__

    def __add__(self, other):
        """Add two sets of metrics together."""
        if not isinstance(other, Metric):
            raise TypeError("Expecting the given other metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(other)))
        return Metric(metrics=self.metrics + other.metrics)

    def __radd__(self, other):
        """Add two sets of metrics together."""
        return self.__add__(other)

    def __iadd__(self, other):
        """Append the other metrics to this one."""
        if not isinstance(other, Metric):
            raise TypeError("Expecting the given other metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(other)))
        self.metrics += other.metrics
