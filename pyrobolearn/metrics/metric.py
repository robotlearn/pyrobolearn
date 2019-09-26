# -*- coding: utf-8 -*-
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
        self.metrics = metrics
        self._data = None

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
        self._check_recursively_metric_type(metrics)
        self._metrics = metrics

    @property
    def data(self):
        """Return the data."""
        if self.metrics:
            return [metric.data for metric in self.metrics]
        return self._get_data()

    ###########
    # Methods #
    ###########

    def _check_recursively_metric_type(self, metrics):
        """Check recursively the metric types."""
        if isinstance(metrics, collections.Iterable):
            for metric in metrics:
                self._check_recursively_metric_type(metric)
        elif not isinstance(metrics, Metric):
            raise TypeError("Expecting the given 'metric' to be an instance of `Metric`, instead got: "
                            "{}".format(type(metrics)))

    def _get_data(self):
        """Return the inner data."""
        return self._data

    def has_inner_metrics(self):
        """Check if the metric has inner metrics."""
        if self.metrics:
            return True
        return False

    def append(self, metric):
        """Append the given metric to the list of metrics."""
        if not isinstance(metric, Metric):
            raise TypeError("Expecting the given metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(metric)))
        self.metrics.append(metric)

    def __call(self, method_name, *args, **kwargs):  # TODO do it in a recursive way
        """Call the method for each metric with the provided arguments."""
        if self.metrics:
            for metric in self.metrics:
                fct = getattr(metric, method_name)
                fct(*args, **kwargs)
        else:
            fct = getattr(self, method_name)
            fct(*args, **kwargs)

    def update(self, *args, **kwargs):
        """Update the metrics."""
        self.__call('_update', *args, **kwargs)

    def _update(self, *args, **kwargs):
        """Update the metric; this has to be implemented in the child class."""
        pass

    def start_algo_update(self, algo):
        """Update each time an algorithm starts."""
        self.__call('_start_algo_update', algo=algo)

    def _start_algo_update(self, algo):
        """Update each time an algorithm starts; this has to be implemented in the child class."""
        pass

    def end_algo_update(self, algo):
        """Update each time an algorithm ends."""
        self.__call('_end_algo_update', algo=algo)

    def _end_algo_update(self, algo):
        """Update each time an algorithm ends; this has to be implemented in the child class."""
        pass

    def start_episode_update(self, episode_idx=None, num_episodes=None):
        """Update each time an episode starts."""
        self.__call('_start_episode_update', episode_idx=episode_idx, num_episodes=num_episodes)

    def _start_episode_update(self, episode_idx=None, num_episodes=None):
        """Update each time an episode starts; this has to be implemented in the child class."""
        pass

    def episode_update(self, episode_idx=None):
        """Update at each episode."""
        self.__call('_episode_update', episode_idx=episode_idx)

    def _episode_update(self, episode_idx=None):
        """Update at each episode; this has to be implemented in the child class."""
        pass

    def end_episode_update(self, episode_idx=None, num_episodes=None):
        """Update each time an episode ends."""
        self.__call('_end_episode_update', episode_idx=episode_idx, num_episodes=num_episodes)

    def _end_episode_update(self, episode_idx=None, num_episodes=None):
        """Update each time an episode ends."""
        pass

    def start_rollout_update(self, rollout_idx=None, num_rollouts=None):
        """Update each time a rollout starts."""
        self.__call('_start_rollout_update', rollout_idx=rollout_idx, num_rollouts=num_rollouts)

    def _start_rollout_update(self, rollout_idx=None, num_rollouts=None):
        """Update each time a rollout starts; this has to be implemented in the child class."""
        pass

    def rollout_update(self, rollout_idx=None):
        """Update at each rollout."""
        self.__call('_rollout_update', rollout_idx=rollout_idx)

    def _rollout_update(self, rollout_idx=None):
        """Update at each rollout; this has to be implemented in the child class."""
        pass

    def end_rollout_update(self, rollout_idx=None, num_rollouts=None):
        """Update each time a rollout ends."""
        self.__call('_end_rollout_update', rollout_idx=rollout_idx, num_rollouts=num_rollouts)

    def _end_rollout_update(self, rollout_idx=None, num_rollouts=None):
        """Update each time a rollout ends; this has to be implemented in the child class."""
        pass

    def start_step_update(self, step_idx=None, num_steps=None):
        """Update each time a step starts."""
        self.__call('_start_step_update', step_idx=step_idx, num_steps=num_steps)

    def _start_step_update(self, step_idx=None, num_steps=None):
        """Update each time a step starts; this has to be implemented in the child class."""
        pass

    def step_update(self, step_idx=None):
        """Update at each time step."""
        self.__call('_step_update', step_idx=step_idx)

    def _step_update(self, step_idx=None):
        """Update at each time step; this has to be implemented in the child class."""
        pass

    def end_step_update(self, step_idx=None, num_steps=None):
        """Update each time a step ends."""
        self.__call('_end_step_update', step_idx=step_idx, num_steps=num_steps)

    def _end_step_update(self, step_idx=None, num_steps=None):
        """Update each time a step ends; this has to be implemented in the child class."""
        pass

    def start_epoch_update(self, epoch_idx=None, num_epochs=None):
        """Update each time an epoch starts."""
        self.__call('_start_epoch_update', epoch_idx=epoch_idx, num_epochs=num_epochs)

    def _start_epoch_update(self, epoch_idx=None, num_epochs=None):
        """Update each time a epoch starts; this has to be implemented in the child class."""
        pass

    def end_epoch_update(self, epoch_idx=None, num_epochs=None):
        """Update each time a epoch ends."""
        self.__call('_end_epoch_update', epoch_idx=epoch_idx, num_epochs=num_epochs)

    def _end_epoch_update(self, epoch_idx=None, num_epochs=None):
        """Update each time a epoch ends; this has to be implemented in the child class."""
        pass

    def start_batch_update(self, batch_idx=None, num_batches=None):
        """Update each time a batch starts."""
        self.__call('_start_batch_update', batch_idx=batch_idx, num_batches=num_batches)

    def _start_batch_update(self, batch_idx=None, num_batches=None):
        """Update each time a batch starts; this has to be implemented in the child class."""
        pass

    def end_batch_update(self, batch_idx=None, num_batches=None):
        """Update each time a batch ends."""
        self.__call('_end_batch_update', batch_idx=batch_idx, num_batches=num_batches)

    def _end_batch_update(self, batch_idx=None, num_batches=None):
        """Update each time a batch ends; this has to be implemented in the child class."""
        pass

    def plot(self, nrows=-1, ncols=-1, block=True, filename=None, _ax=None):
        """
        Plot the metric(s).

        Args:
            nrows (int): number of rows in the subplot.
            ncols (int): number of columns in the subplot.
            block (bool): if True, it will block when showing the graphs.
            filename (str, None): if a string is given, it will save the plot in the given filename.
            _ax (plt.Axes, None): axis to plot the figure. Do not use this parameter, this is used internally in a
                recursive way.

        Returns:
            matplotlib.figure.Figure: figure
            np.array of matplotlib.axes._subplots.AxesSubplot: axes
        """
        if not self.metrics and _ax is not None:
            self._plot(ax=_ax)

        else:
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
            for i, metric in enumerate(metrics):
                metric.plot(_ax=axes[i])

            # tight the layout
            fig.tight_layout()

            # save figure if specified
            if filename is not None:
                fig.savefig(filename)

            # show plot
            plt.show(block=block)

            # return figure and axes
            return fig, axes

    def _plot(self, ax):
        """
        Plot the metric in the given axis. This has to be implemented in the child classes.
        """
        pass

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
            return ' + '.join([str(metric) for metric in self.metrics])
        return self.__class__.__name__

    def __add__(self, other):
        """Add two sets of metrics together; they will be in the same figure but in different subplots."""
        if not isinstance(other, Metric):
            raise TypeError("Expecting the given other metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(other)))
        if self.metrics:
            if other.metrics:
                return Metric(metrics=self.metrics + other.metrics)
            return Metric(metrics=self.metrics + [other])
        if other.metrics:
            return Metric(metrics=[self] + other.metrics)
        return Metric(metrics=[self, other])

    def __radd__(self, other):
        """Add two sets of metrics together."""
        return self.__add__(other)

    def __iadd__(self, other):
        """Append the other metrics to this one."""
        if not isinstance(other, Metric):
            raise TypeError("Expecting the given other metric to be an instance of `Metric`, but got instead: "
                            "{}".format(type(other)))
        self.metrics += other.metrics
