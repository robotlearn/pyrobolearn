#!/usr/bin/env python
"""Provides the experience replay (ER) storage.

References:
    [1] "Reinforcement Learning for robots using neural networks", Lin, 1993
    [2] "Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013
"""

import random
import torch

from pyrobolearn.storages.storage import Storage, ListStorage


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ExperienceReplay(Storage):
    r"""Experience replay storage

    The experience replay storage returns a transition tuple :math:`(s_t, a_t, s_{t+1}, r_t, d)`, where is :math:`s_t`
    is the state at time :math:`t`, :math:`a_t` is the action outputted by the policy in response to the state
    :math:`s_t`, :math:`s_{t+1}` is the next state returned by the environment due to the policy's action :math:`a_t`
    and the current state :math:`s_t`, :math:`r_t` is the reward signal returned by the environment, and :math:`d`
    is a boolean value that specifies if the task is over or not (i.e. if it has failed or succeeded).

    The experience replay storage is often used in conjunction with off-policy RL algorithms.

    The following code is inspired by [3] but modified such that it uses a PyTorch list storage.

    References:
        [1] "Reinforcement Learning for robots using neural networks", Lin, 1993
        [2] "Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013
    """

    def __init__(self, capacity=10000, device=None, dtype=torch.float):
        """
        Initialize the Experience Replay Storage.

        Args:
            capacity (int): maximum size of the experience replay storage.
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
        """
        super(ExperienceReplay, self).__init__(device, dtype)
        self.capacity = capacity
        self.memory = ListStorage(device=device, dtype=dtype)
        self.position = 0

    ##############
    # Properties #
    ##############

    @property
    def device(self):
        """Return the memory's device instance."""
        return self.memory.device

    @device.setter
    def device(self, device):
        """Set the memory's device instance."""
        self.memory.device = device

    @property
    def dtype(self):
        """Return the memory's data type."""
        return self.memory.dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set the memory's data type."""
        self.memory.dtype = dtype

    ###########
    # Methods #
    ###########

    def push(self, *items):
        r"""
        Push new transition :math:`(s_t, a_t, s_{t+1}, r_t, d, \gamma)` in the experience replay.

        Args:
            *items (list, tuple of torch.Tensor): transition tuple
        """
        # add new item or update previous one
        if len(self.memory) < self.capacity:
            self.memory.append(items)
        else:
            self.memory[self.position] = items

        # update head position (cyclic)
        self.position = (self.position + 1) % self.capacity

    def to(self, device=None, dtype=None):
        """
        Put all the tensors to the specified device and convert them to the specified data type.

        Args:
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
        """
        self.memory.to(device=device, dtype=dtype)

    def sample(self, batch_size):
        """Sample uniformly a batch from the experience replay."""
        return random.sample(self, batch_size)


# alias
ER = ExperienceReplay
