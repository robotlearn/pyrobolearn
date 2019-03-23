#!/usr/bin/env python
"""Provide some reward processors.

It processes the rewards before returning them; this can be useful to standardize, normalize, center them for instance.
"""

import numpy as np
from reward import Reward

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RewardProcessor(Reward):
    r"""Reward Processor

    Wraps the reward and process it. It also acts as a memory of the last received reward signal.

    Examples:
        reward = Reward1() + Reward2()
        reward = RewardProcessor(reward, <args>)
    """

    def __init__(self, reward):
        super(RewardProcessor, self).__init__()
        self.reward = reward
        self.value = 0

    def compute(self):
        self.value = self.reward()
        return self.value


class ShiftRewardProcessor(RewardProcessor):
    r"""Shift Reward Processor

    Shift the reward by the given amount; that is, it returned: :math:`\hat{r} = r + x` where :math:`x` is the
    specified amount to shift the original reward.
    """

    def __init__(self, reward, x):
        """
        Initialize the Center Reward Processor.

        Args:
            reward (Reward): Reward instance to process.
            x (int, float): amount to be shifted.
        """
        super(ShiftRewardProcessor, self).__init__(reward)
        self.x = x

    def compute(self):
        reward = self.reward()
        self.value = reward + self.x
        return self.value


class ClipRewardProcessor(RewardProcessor):
    r"""Clip Reward Processor

    Processor that clips the given reward to be between [low, high], where `low` and `high` are respectively the
    specified lower and higher bound.
    """

    def __init__(self, reward, low=-10, high=10):
        """
        Initialize the Clip processor.

        Args:
            reward (Reward): Reward instance to process.
            low (int, float): lower bound
            high (int, float): higher bound
        """
        super(ClipRewardProcessor, self).__init__(reward)
        self.low = low
        self.high = high

    def compute(self):
        reward = self.reward()
        self.value = np.clip(reward, self.low, self.high)
        return self.value


class CenterRewardProcessor(RewardProcessor):
    r"""Center Reward Processor

    Center the reward using the running mean.
    """

    def __init__(self, reward):
        super(CenterRewardProcessor, self).__init__(reward)
        self.mean = 0
        self.N = 0

    def reset(self):
        self.mean = 0
        self.N = 0
        self.reward.reset()

    def compute(self):
        reward = self.reward()
        # update the mean
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * reward
        self.N += 1
        # center reward
        self.value = self.value - self.mean
        return self.value


class NormalizeRewardProcessor(RewardProcessor):
    r"""Normalize Reward Processor

    Normalize the reward such that it is between 0 and 1. That is, it returned
    :math:`\hat{r} = \frac{r - r_{min}}{r_{max} - r_{min}}`, where :math:`r \in [r_{min}, r_{max}]`.

    Warnings: the first returned reward will be 0.
    """

    def __init__(self, reward):
        super(NormalizeRewardProcessor, self).__init__(reward)
        self.min = np.infty
        self.max = -np.infty

    def reset(self):
        self.min = np.infty
        self.max = -np.infty
        self.reward.reset()

    def compute(self):
        reward = self.reward()
        self.min = np.minimum(reward, self.min)
        self.max = np.maximum(reward, self.max)
        den = self.max - self.min
        if den == 0:
            den = 1.
        self.value = (reward - self.min) / den
        return self.value


class StandardizeRewardProcessor(RewardProcessor):
    r"""Standardize Reward Processor

    Standardize the reward such that it returns :math:`\hat{r} = \frac{r - \mu}{\sigma}` where  :math:`\mu` is the
    running mean, and :math:`\sigma` is the running standard deviation. The returned reward will have a mean of 0
    and standard deviation of 1.
    """

    def __init__(self, reward, epsilon=1.e-4, center=True):
        super(StandardizeRewardProcessor, self).__init__(reward)
        self.eps = epsilon
        self.mean = 0
        self.var = 1
        self.N = 0
        self.center = center

    def reset(self):
        self.mean = 0
        self.var = 1
        self.N = 0
        self.reward.reset()

    def compute(self):
        reward = self.reward()

        # update the mean
        old_mean = self.mean
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * reward

        # update the var / stddev
        self.var = self.N / (self.N + 1) * self.var + \
                   1. / (self.N + 1) * (self.value - old_mean) * (self.value - self.mean)
        std = np.sqrt(self.var)

        # update total number of data points
        self.N += 1

        # standardize the reward
        if self.center:
            self.value = (reward - self.mean) / (std + self.eps)
        else:
            self.value = reward / (std + self.eps)
        return self.value


class GammaAccumulatedRewardProcessor(RewardProcessor):
    r"""Gamma reward processor

    It will return the accumulated reward until now: :math:`R = \sum_{t'=0}^t \gamma^{t'} r_{t'}`.
    """

    def __init__(self, reward, gamma=0.99):
        super(GammaAccumulatedRewardProcessor, self).__init__(reward)
        self.gamma = gamma
        self.ret = 0.

    def reset(self):
        self.ret = 0.
        self.reward.reset()

    def compute(self):
        reward = self.reward()
        self.ret = reward + self.gamma * self.ret
        return self.ret


class GammaStandardizeRewardProcessor(RewardProcessor):
    r"""Gamma Standardize Reward Processor

    References:
        [1] https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
    """

    def __init__(self, reward, gamma=0.99, epsilon=1.e-4):
        super(GammaStandardizeRewardProcessor, self).__init__(reward)
        self.gamma = gamma
        self.eps = epsilon
        self.ret = 0
        self.mean = 0
        self.var = 1
        self.N = 0

    def reset(self):
        self.ret = 0
        self.mean = 0
        self.var = 1
        self.N = 0
        self.reward.reset()

    def compute(self):
        reward = self.reward()

        # update return
        self.ret = reward + self.gamma * self.ret

        # update the return mean
        old_mean = self.mean
        self.mean = self.N / (self.N + 1.) * self.mean + 1. / (self.N + 1) * self.ret

        # update the return variance
        self.var = self.N / (self.N + 1) * self.var + 1. / (self.N + 1) * (self.ret - old_mean) * (self.ret - self.mean)
        std = np.sqrt(self.var)

        # update total number of data points
        self.N += 1

        self.value = reward / (std + self.eps)
        return self.value


class ScaleRewardProcessor(RewardProcessor):
    r"""Scale Reward Processor

    Processor that scales the reward x which is between [x1, x2] to the output y which is between [y1, y2].
    """

    def __init__(self, reward, x1, x2, y1, y2):
        """
        Initialize the scale reward processor

        Args:
            reward (Reward): reward function to process.
            x1 (int, float): lower bound of the original reward
            x2 (int, float): upper bound of the original reward
            y1 (int, float): lower bound of the final reward
            y2 (int, float): upper bound of the final reward
        """
        super(ScaleRewardProcessor, self).__init__(reward)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.ratio = (self.y2 - self.y1) / (self.x2 - self.x1)

    @convert_numpy
    def compute(self):
        reward = self.reward()
        self.value = self.y1 + (reward - self.x1) * self.ratio
        return self.value
