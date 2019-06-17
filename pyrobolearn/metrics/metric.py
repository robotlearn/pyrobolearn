#!/usr/bin/env python
"""Defines the various metrics used in different learning paradigms.

Dependencies:
- `pyrobolearn.tasks`
"""

import collections
import matplotlib.pyplot as plt


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
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

    It notably contains the functionalities to evaluate a certain task using the metric, and different to plot them.
    """

    def __init__(self, metrics=None):
        """
        Initialize the metric object.

        Args:
            metrics (None, Metric, list of Metric): inner metric objects.
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

    def append(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def _plot(self, ax=None, filename=None):
        """
        Plot the metric. This has to be implemented in the child classes.

        Args:
            ax (plt.Axes): axis to plot the figure.
            filename (str, None): if a string is given, it will save the plot in the given filename.

        Returns:
            plt.Axes: ax
        """
        pass

    def plot(self, ax=None, block=False, filename=None, subplots=()):
        """
        Plot the metric.

        Args:
            ax (plt.Axes): axis to plot the figure.
            block (bool): if True, it will block when showing the graphs.
            filename (str, None): if a string is given, it will save the plot in the given filename.
        """
        # if multiple metrics
        if self.metrics:

            # if we want to use subplots
            if len(subplots) > 0:
                pass

            # if we just want multiple figures
            for metric in self.metrics:
                metric._plot(ax=ax, filename=filename)

            plt.show(block=block)
        else:
            self._plot(ax=ax, filename=filename)
            plt.show(block=block)

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string of the object."""
        if self.metrics:
            return ' + '.join(self.metrics)
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the object."""
        if self.metrics:
            return ' + '.join(self.metrics)
        return self.__class__.__name__
