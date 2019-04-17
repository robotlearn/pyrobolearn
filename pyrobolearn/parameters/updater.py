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
    r"""Parameter updater (abstract) class.

    The parameter updater represents a scheme on how to update certain parameters (int, float, torch.nn.Module,
    torch.tensor, np.array, etc).
    """

    def __init__(self, current=None, target=None, sleep_count=1):
        """
        Initialize the parameter updater.

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s).
            sleep_count (int): number of ticks to sleep
        """
        self.sleep_count = 1 if sleep_count <= 0 else int(sleep_count)
        self.counter = 0
        self.current = current
        self.target = target

    def _compute(self, current, target):
        """
        Inner update that needs to be inherited in all the chid classes. It updates the target parameter(s) based on
        the current parameter(s). This is an in-place operation; it modifies the given target.

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module): target parameter(s) to be modified based on
            the current parameter(s).

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        pass

    def compute(self, current=None, target=None):
        """
        Update the target parameter(s) based on the current parameter(s).

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s). If None, it
                will be set to the current parameter(s) given at the initialization.
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s). If None, it will be set to the current parameters.

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        # if time to update the parameters
        if (self.counter % self.sleep_count) == 0:

            # if the current parameters is None, take the ones given at the initialization
            if current is None:
                current = self.current
                if current is None:
                    raise ValueError("Expecting to be given current parameters, instead got None.")

            # if the target is None, set it to be the current parameters.
            if target is None:
                target = current

            # the target and current needs to be of the same type (except if current is a float or int)
            if not isinstance(current, (int, float)) and type(target) != type(current):
                raise TypeError("The given target and current parameters are of different types: "
                                "type(target)={} and type(current)={}".format(type(target), type(current)))

            # inner computation
            self.target = self._compute(current, target)

        # increment counter
        self.counter += 1

        # return the new target parameter(s).
        return self.target

    def __call__(self, current=None, target=None):
        """Alias to :func:`compute` method. See doc for :func:`compute`."""
        return self.compute(current=current, target=target)


class CopyParameter(ParameterUpdater):
    r"""Copy parameters

    Copy the parameters every `sleep_count` times it is called.
    """

    def __init__(self, current=None, target=None, sleep_count=1):
        """
        Initialize the copy parameter updater.

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s).
            sleep_count (int): number of ticks to sleep
        """
        super(CopyParameter, self).__init__(current, target, sleep_count)

    def _compute(self, current, target):
        """
        Updates the target parameter(s) based on the current parameter(s).

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module): target parameter(s) to be modified based on
            the current parameter(s).

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    p1.data.copy_(p2.data)
        elif isinstance(target, torch.Tensor):
            target.data.copy_(current.data)
        elif isinstance(target, np.ndarray):
            target[:] = current
        else:
            target = current
        return target


class LinearDecay(ParameterUpdater):
    r"""Linear Decay

    A linear model can be described mathematically :math:`y = ax + b`, where :math:`a` is the slope, and :math:`b`
    is the intercept. The returned value will be :math:`y_{t+1} = y_t + a * dx`, where :math:`dx` is the integration
    step size.
    """

    def __init__(self, slope, step=0.01, end=None, current=None, target=None, sleep_count=1):
        """
        Initialize the linear decay parameter updater.

        Args:
            slope (float): slope.
            step (float): integration step size.
            end (float, None): end value. If the slope is negative (resp. positive), this is the minimum (resp.
                maximum) value it can take.
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s).
            sleep_count (int): number of ticks to sleep
        """
        super(LinearDecay, self).__init__(current, target, sleep_count)
        self.slope = slope
        self.dt = step
        if end is None:
            end = -np.infty if slope < 0 else np.infty
        self.end = end

    def _compute(self, current, target):
        """
        Updates the target parameter(s) based on the current parameter(s).

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module): target parameter(s) to be modified based on
            the current parameter(s).

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    data = p2.data + self.slope * self.dt
                    if self.slope < 0:
                        torch.clamp_min_(data, self.end)
                    else:
                        torch.clamp_max_(data, self.end)
                    p1.data.copy_(data)
        elif isinstance(target, torch.Tensor):
            data = current.data + self.slope * self.dt
            if self.slope < 0:
                torch.clamp_min_(data, self.end)
            else:
                torch.clamp_max_(data, self.end)
            target.data.copy_(data)
        elif isinstance(target, np.ndarray):
            target[:] = current + self.slope * self.dt
            if (self.slope < 0 and target < self.end) or (self.slope > 0 and target > self.end):
                target[:] = self.end
        else:
            target = current + self.slope * self.dt
            if (self.slope < 0 and target < self.end) or (self.slope > 0 and target > self.end):
                target = self.end
        return target


class ExponentialDecay(ParameterUpdater):
    r"""Exponential decay

    Compute: :math:`y = a \exp{b x}`, and thus :math:`y_{t+1} = y_{t} + b y_{t} dx`, where :math:`b` is the speed
    at which the exponential converges to 0 if negative, and diverges to infinity if positive.
    """

    def __init__(self, speed, step=0.01, end=None, current=None, target=None, sleep_count=1):
        """
        Initialize the exponential decay parameter updater.

        Args:
            speed (float): speed.
            step (float): integration step size.
            end (float, None): end value. If the speed is negative (resp. positive), this is the minimum (resp.
                maximum) value it can take.
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s).
            sleep_count (int): number of ticks to sleep
        """
        super(ExponentialDecay, self).__init__(current, target, sleep_count)
        self.speed = speed
        self.dt = step
        if end is None:
            end = -np.infty if speed < 0 else np.infty
        self.end = end

    def _compute(self, current, target):
        """
        Updates the target parameter(s) based on the current parameter(s).

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module): target parameter(s) to be modified based on
            the current parameter(s).

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    data = p2.data + self.speed * p2.data * self.dt
                    if self.speed < 0:
                        torch.clamp_min_(data, self.end)
                    else:
                        torch.clamp_max_(data, self.end)
                    p1.data.copy_(data)
        elif isinstance(target, torch.Tensor):
            data = current.data + self.speed * current.data * self.dt
            if self.speed < 0:
                torch.clamp_min_(data, self.end)
            else:
                torch.clamp_max_(data, self.end)
            target.data.copy_(data)
        elif isinstance(target, np.ndarray):
            target[:] = current + self.speed * current * self.dt
            if (self.speed < 0 and target < self.end) or (self.speed > 0 and target > self.end):
                target[:] = self.end
        else:
            target = current + self.speed * current * self.dt
            if (self.speed < 0 and target < self.end) or (self.speed > 0 and target > self.end):
                target = self.end
        return target


class PolyakAveraging(ParameterUpdater):
    r"""Polyak Averaging

    Compute: :math:`y^* = \rho y^* + (1 - \rho) y`, where :math:`\rho` is a parameter which is between 0 and 1,
    :math:`y^*` is the target, and :math:`y` is the current value.
    """

    def __init__(self, rho=0., current=None, target=None, sleep_count=1):
        """
        Initialize the polyak averaging parameter updater.

        Args:
            rho (float): float value between 0 and 1. If 1, it will let the target parameter(s) unchanged, if 0 it
                will just copy the current parameter(s).
            current (int, float, torch.tensor, np.array, torch.nn.Module, None): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module, None): target parameter(s) to be modified
                based on the current parameter(s).
            sleep_count (int): number of ticks to sleep
        """
        super(PolyakAveraging, self).__init__(sleep_count)
        if not (0. <= rho <= 1.):
            raise ValueError("Expecting the given 'rho' to be a float between 0 and 1, instead got: {}".format(rho))
        self.rho = rho

    def _compute(self, current, target):
        """
        Updates the target parameter(s) based on the current parameter(s).

        Args:
            current (int, float, torch.tensor, np.array, torch.nn.Module): current parameter(s).
            target (int, float, torch.tensor, np.array, torch.nn.Module): target parameter(s) to be modified based on
            the current parameter(s).

        Returns:
            int, float, torch.tensor, np.array, torch.nn.Module: updated target parameter(s).
        """
        if isinstance(target, torch.nn.Module):
            if isinstance(current, torch.nn.Module):
                for p1, p2 in zip(target.parameters(), current.parameters()):
                    p1.data.copy_(self.rho * p1.data + (1 - self.rho) * p2.data)
        elif isinstance(target, torch.Tensor):
            target.data.copy_(self.rho * target.data + (1 - self.rho) * current.data)
        elif isinstance(target, np.ndarray):
            target[:] = self.rho * target + (1 - self.rho) * current
        else:
            target = self.rho * target + (1 - self.rho) * current
        return target
