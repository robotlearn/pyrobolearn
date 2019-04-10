#!/usr/bin/env python
r"""Provide the various update schemes to update optimization parameters and model parameters.

These parameters include parameters of a learning model, simple weight numbers (int or float) used in losses/rewards,
and others.
"""

import numpy as np
import torch


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ParameterUpdater(object):

    def __init__(self, sleep_count=1):
        """
        Initialize the parameter updater.

        Args:
            sleep_count (int): number of ticks to sleep
        """
        self.sleep_count = 1 if sleep_count <= 0 else int(sleep_count)
        self.counter = 0
        self.target = None

    def _compute(self, current, target):
        pass

    def compute(self, current, target=None):
        if (self.counter % self.sleep_count) == 0:
            if target is None:
                target = current
            self._compute(current, target)
        self.counter += 1
        return self.target

    def __call__(self, current, target):
        return self.compute(current, target)


class CopyParameter(ParameterUpdater):
    r"""Copy parameters

    Copy the parameters every `sleep_count` times it is called.
    """

    def __init__(self, sleep_count=1):
        """
        Initialize the copy parameter updater.

        Args:
            sleep_count (int): number of ticks to sleep
        """
        super(CopyParameter, self).__init__(sleep_count)

    def _compute(self, current, target):
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    p1.data.copy_(p2.data)
        else:
            target = current
        self.target = target


class LinearDecay(ParameterUpdater):
    r"""Linear Decay

    A linear model can be described mathematically :math:`y = ax + b`, where :math:`a` is the slope, and :math:`b`
    is the intercept. The returned value will be :math:`y_{t+1} = y_t + a * dx`, where :math:`dx` is the integration
    step size.
    """

    def __init__(self, slope, step=0.01, end=None, sleep_count=1):
        """
        Initialize the linear decay parameter updater.

        Args:
            slope (float): slope.
            step (float): integration step size.
            end (float, None): end value. If the slope is negative (resp. positive), this is the minimum (resp.
                maximum) value it can take.
            sleep_count (int): number of ticks to sleep
        """
        super(LinearDecay, self).__init__(sleep_count)
        self.slope = slope
        self.dt = step
        if end is None:
            end = -np.infty if slope < 0 else np.infty
        self.end = end

    def _compute(self, current, target):
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    data = p2.data + self.slope * torch.ones_like(p2.data) * self.dt
                    if self.slope < 0:
                        torch.clamp_min_(data, self.end)
                    else:
                        torch.clamp_max_(data, self.end)
                    p1.data.copy_(data)
        else:
            target = current + self.slope * self.dt
            if (self.slope < 0 and target < self.end) or (self.slope > 0 and target > self.end):
                target = self.end
        self.target = target


class ExponentialDecay(ParameterUpdater):
    r"""Exponential decay

    Compute: :math:`y = a \exp{b x}`, and thus :math:`y_{t+1} = y_{t} + b y_{t} dx`, where :math:`b` is the speed
    at which the exponential converges to 0 if negative, and diverges to infinity if positive.
    """

    def __int__(self, speed, step=0.01, end=None, sleep_count=1):
        """
        Initialize the exponential decay parameter updater.

        Args:
            speed (float): speed.
            step (float): integration step size.
            end (float, None): end value. If the speed is negative (resp. positive), this is the minimum (resp.
                maximum) value it can take.
            sleep_count (int): number of ticks to sleep
        """
        self.speed = speed
        self.dt = step
        if end is None:
            end = -np.infty if speed < 0 else np.infty
        self.end = end

    def _compute(self, current, target):
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    data = p2.data + self.speed * p2.data * self.dt
                    if self.speed < 0:
                        torch.clamp_min_(data, self.end)
                    else:
                        torch.clamp_max_(data, self.end)
                    p1.data.copy_(data)
        else:
            target = current + self.speed * current * self.dt
            if (self.speed < 0 and target < self.end) or (self.speed > 0 and target > self.end):
                target = self.end
        self.target = target


class PolyakAveraging(ParameterUpdater):
    r"""Polyak Averaging

    Compute: :math:`y^* = \rho y^* + (1 - \rho) y`, where :math:`\rho` is a parameter which is between 0 and 1,
    :math:`y^*` is the target, and :math:`y` is the current value.
    """

    def __init__(self, rho=0., sleep_count=1):
        """
        Initialize the polyak averaging parameter updater.

        Args:
            rho (float): float value between 0 and 1. If 1, it won't do anything, if 0 it will
            sleep_count (int): number of ticks to sleep
        """
        super(PolyakAveraging, self).__init__(sleep_count)
        if not (0. <= rho <= 1.):
            raise ValueError("Expecting the given 'rho' to be a float between 0 and 1, instead got: {}".format(rho))
        self.rho = rho

    def _compute(self, current, target):
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    p1.data.copy_(self.rho * p1.data + (1 - self.rho) * p2.data)
        else:
            target = self.rho * target + (1 - self.rho) * current
        self.target = target
