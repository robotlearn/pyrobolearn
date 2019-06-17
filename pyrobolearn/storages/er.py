#!/usr/bin/env python
"""Provides the experience replay (ER) storage.

References:
    [1] "Reinforcement Learning for robots using neural networks", Lin, 1993
    [2] "Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013
"""

import random
import numpy as np
import torch

from pyrobolearn.storages.storage import Storage, ListStorage, Batch, DictStorage


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# class ExperienceReplay(Storage):
#     r"""Experience replay storage
#
#     The experience replay storage returns a transition tuple :math:`(s_t, a_t, s_{t+1}, r_t, d)`, where is :math:`s_t`
#     is the state at time :math:`t`, :math:`a_t` is the action outputted by the policy in response to the state
#     :math:`s_t`, :math:`s_{t+1}` is the next state returned by the environment due to the policy's action :math:`a_t`
#     and the current state :math:`s_t`, :math:`r_t` is the reward signal returned by the environment, and :math:`d`
#     is a boolean value that specifies if the task is over or not (i.e. if it has failed or succeeded).
#
#     The experience replay storage is often used in conjunction with off-policy RL algorithms.
#
#     The following code is inspired by [3] but modified such that it uses a PyTorch list storage.
#
#     References:
#         [1] "Reinforcement Learning for robots using neural networks", Lin, 1993
#         [2] "Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013
#     """
#
#     def __init__(self, capacity=10000, device=None, dtype=torch.float):
#         """
#         Initialize the Experience Replay Storage.
#
#         Args:
#             capacity (int): maximum size of the experience replay storage.
#             device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
#                `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
#                to which the tensor is allocated.
#             dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
#                 the original dtype
#         """
#         super(ExperienceReplay, self).__init__(device, dtype)
#         if capacity <= 0:
#             raise ValueError("Expecting the capacity to be bigger than 0, instead got: {}".format(capacity))
#         self.capacity = capacity
#         self.memory = ListStorage(device=device, dtype=dtype)
#         self.position = 0
#
#     ##############
#     # Properties #
#     ##############
#
#     @property
#     def device(self):
#         """Return the memory's device instance."""
#         return self.memory.device
#
#     @device.setter
#     def device(self, device):
#         """Set the memory's device instance."""
#         self.memory.device = device
#
#     @property
#     def dtype(self):
#         """Return the memory's data type."""
#         return self.memory.dtype
#
#     @dtype.setter
#     def dtype(self, dtype):
#         """Set the memory's data type."""
#         self.memory.dtype = dtype
#
#     @property
#     def size(self):
#         """Return the size of the experience replay storage, i.e. the number of transition tuples."""
#         return len(self.memory)
#
#     ###########
#     # Methods #
#     ###########
#
#     def push(self, *items):
#         r"""
#         Push new transition :math:`(s_t, a_t, s_{t+1}, r_t, d)` in the experience replay.
#
#         Args:
#             *items (list, tuple of torch.Tensor): transition tuple
#         """
#         # add new item or update previous one
#         if len(self.memory) < self.capacity:
#             self.memory.append(items)
#         else:
#             self.memory[self.position] = items
#
#         # update head position (cyclic)
#         self.position = (self.position + 1) % self.capacity
#
#     def to(self, device=None, dtype=None):
#         """
#         Put all the tensors to the specified device and convert them to the specified data type.
#
#         Args:
#             device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
#                `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
#                to which the tensor is allocated.
#             dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
#                 the original dtype
#         """
#         self.memory.to(device=device, dtype=dtype)
#
#     def sample(self, batch_size):
#         """Sample uniformly a batch from the experience replay."""
#         return random.sample(self, batch_size)
#
#     def get_batch(self, indices):
#         """Return a batch of the transitions in the form of a `DictStorage`.
#
#         Args:
#             indices (list of int): indices. Each index must be between 0 and `self.size`.
#
#         Returns:
#             DictStorage / Batch: batch containing a part of the storage. Variables such as `states`, `actions`,
#                 `rewards`, `next_states`, `masks`, and others can be accessed from the object.
#         """
#         # create batch
#         batch = {}
#
#         # check indices
#         indices = np.array(indices)
#         indices = indices[indices < self.size]
#
#         for idx in indices:
#             transition = self.memory[idx]
#             batch.setdefault('states', []).append(transition[0])
#             batch.setdefault('actions', []).append(transition[1])
#             batch.setdefault('next_states', []).append(transition[2])
#             batch.setdefault('rewards', []).append(transition[3])
#             batch.setdefault('masks', []).append(transition[4])
#
#         return Batch(batch, device=self.device, dtype=self.dtype)
#
#
# # alias
# ER = ExperienceReplay


class ExperienceReplay(DictStorage):  # ExperienceReplayStorage(DictStorage):
    r"""Experience Replay Storage

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

    def __init__(self, state_shapes, action_shapes, capacity=10000, *args, **kwargs):
        """
        Initialize the experience replay storage.

        Args:
            state_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an observation/state.
            action_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an action.
            capacity (int): maximum size of the experience replay storage.
        """
        # recurrent_hidden_state_size (int): size of the internal state
        print("\nStorage: state shape: {}".format(state_shapes))
        print("Storage: action shape: {}".format(action_shapes))
        super(ExperienceReplay, self).__init__()
        self.position = 0
        if capacity <= 0:
            raise ValueError("Expecting the capacity to be bigger than 0, instead got: {}".format(capacity))
        self.capacity = capacity
        self.full = False
        self.init(state_shapes, action_shapes)

    ##############
    # Properties #
    ##############

    @property
    def size(self):
        """Return the size of the experience replay storage."""
        # if self.full:
        #     return self.capacity
        # return self.position
        return self.capacity

    ###########
    # Methods #
    ###########

    def create_new_entry(self, key, shapes, dtype=torch.dtype):
        """Create a new entry (=tensor) in the experience replay storage dictionary. The tensor will have the
        dimension (capacity, *shape) for each shape in shapes, and will be initialized to zero.
        The tensor will also have the same type than the other tensors and will be sent to the correct device.

        Args:
            key (str, object): key of the dictionary.
            shapes (list of tuple of int, tuple of int, int): (list of) shape(s) of the tensor(s).
            dtype (torch.dtype, np.generic, object): specify if we want to allocate a `torch.Tensor`, or a numpy array.
                If dtype == torch.dtype, then it will be set to dtype = self.dtype.
        """
        # allocate the new tensor

        # convert torch type if necessary
        if dtype == torch.dtype:
            dtype = self.dtype

        # if we have a list of shapes
        if isinstance(shapes, list):
            if isinstance(dtype, torch.dtype):
                self[key] = [torch.zeros(self.capacity, *shape).to(device=self.device, dtype=dtype)
                             for shape in shapes]
            else:  # numpy array
                self[key] = [np.zeros((self.capacity,) + shape, dtype=dtype) for shape in shapes]

        # if the 'shapes' is a tuple
        elif isinstance(shapes, tuple):
            if isinstance(dtype, torch.dtype):
                self[key] = torch.zeros(self.capacity, *shapes).to(device=self.device, dtype=dtype)
            else:
                self[key] = np.zeros((self.capacity,) + shapes, dtype=dtype)

        # if the 'shapes' is an int
        elif isinstance(shapes, int):
            if isinstance(dtype, torch.dtype):
                self[key] = torch.zeros(self.capacity, shapes).to(device=self.device, dtype=dtype)
            else:
                self[key] = np.zeros((self.capacity, shapes), dtype=dtype)

        else:
            raise TypeError("Expecting the given shapes {} to be a list of tuple of int, a tuple of int, or an int, "
                            "instead got: {}".format({}, type(shapes)))

    def init(self, state_shapes, action_shapes):
        """
        Initialize the experience replay storage by allocating the appropriate tensors for the observations (states),
        actions, next states, rewards, and masks.

        Args:
            state_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an observation/state.
            action_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an action.
        """
        # clear itself: remove all items from the DictStorage, and reset all variables
        self.reset()

        # allocate space for observations / states
        if not isinstance(state_shapes, list):
            state_shapes = [state_shapes]
        self.create_new_entry('states', shapes=state_shapes)

        # allocate space for actions
        if not isinstance(action_shapes, list):
            action_shapes = [action_shapes]
        self.create_new_entry('actions', shapes=action_shapes)

        # allocate space for next observations / states  # TODO: improve this, don't copy, just keep reference
        if not isinstance(state_shapes, list):
            state_shapes = [state_shapes]
        self.create_new_entry('next_states', shapes=state_shapes)

        # allocate space for rewards
        self.create_new_entry('rewards', shapes=1)

        # allocate space for the masks
        self.create_new_entry('masks', shapes=1)

        # space for log probabilities on policy, distributions, scalar values from value functions,
        # recurrent hidden states, and others have to be allocated outside the class

    def reset(self, *args, **kwargs):
        """Reset the experience replay storage."""
        pass

    def clear(self):
        """Clear the experience replay storage."""
        self.full = False
        # super(ExperienceReplayStorage, self).clear()
        super(ExperienceReplay, self).clear()

    def insert(self, states, actions, next_states, reward, mask, **kwargs):
        """
        Insert the given parameters into the storage.

        Args:
            states (torch.Tensor, list of torch.Tensor): (list of) state(s) / observation(s).
            actions (torch.Tensor, list of torch.Tensor): (list of) action(s).
            next_states (torch.Tensor, list of torch.Tensor): (list of) next state(s) / observation(s).
            reward (float, int, torch.Tensor): reward value.
            mask (float, int, torch.Tensor): masks. They are set to zeros after an episode has terminated.
            **kwargs (dict): kwargs
        """
        print("ER - insert state: {}".format(states))
        print("ER - insert action: {}".format(actions))
        print("ER - insert next state: {}".format(next_states))
        print("ER - insert reward: {}".format(reward))
        print("ER - insert mask: {}".format(mask))

        # check given states/observations and actions
        if not isinstance(states, list):
            states = [states]
        if not isinstance(actions, list):
            actions = [actions]
        if not isinstance(next_states, list):
            next_states = [next_states]

        # insert each states / actions / next states
        for state, storage in zip(states, self.states):
            storage[self.position].copy_(self._convert_to_tensor(state))
        for action, storage in zip(actions, self.actions):
            storage[self.position].copy_(self._convert_to_tensor(action))
        for next_state, storage in zip(next_states, self.next_states):
            storage[self.position].copy_(self._convert_to_tensor(next_state))

        # insert reward and mask
        self.rewards[self.position].copy_(self._convert_to_tensor(reward))
        self.masks[self.position].copy_(self._convert_to_tensor(mask))

        # update head position (cyclic)
        self.position += 1
        if self.position >= self.capacity:
            self.full = True
            self.position = self.position % self.capacity

    def add_trajectory(self, trajectory, **kwargs):
        r"""
        Add a trajectory/rollout [(s_t, a_t, s_{t+1}, r_t, d_t)]_{t=1}^T in the storage. This calls in for-loop the
        `insert` method.

        Args:
            trajectory (list of dict): trajectory represented as a list of dictionaries where each dictionary contains
                a transition tuple (s_t, a_t, s_{t+1}, r_t, d_t), and thus has at least the following key: `states`,
                `actions`, `next_states`, `reward`, `mask`.
            **kwargs (dict): kwargs
        """
        # insert each step in the trajectory into the storage
        for step in trajectory:
            self.insert(**step)

    def get_batch(self, indices):
        """Return a batch of the experience replay storage in the form of a `DictStorage`.

        Args:
            indices (list of int): indices. Each index must be between 0 and `self.size`.

        Returns:
            DictStorage / Batch: batch containing a part of the storage. Variables such as `states`, `actions`,
                `next_states`, `rewards`, and `masks`.
        """
        batch = {}

        # go through each attribute in the  and sample from the tensors
        for key, value in self.iteritems():
            if isinstance(value, list):  # value = list of tensors
                batch[key] = [val[indices] for val in value]
            else:  # value = tensor
                batch[key] = value[indices]

        # return batch (which is given to the updater (and loss))
        return Batch(batch, device=self.device, dtype=self.dtype)

    #############
    # Operators #
    #############

    def __setitem__(self, key, value):
        """Add the new value in the dictionary. Key must be strings or objects."""
        # if not isinstance(key, str):
        #     raise TypeError("The experience replay storage only accepts key as strings! Instead got: {} with type "
        #                     "{}".format(key, type(key)))
        # super(ExperienceReplayStorage, self).__setitem__(key, value)
        super(ExperienceReplay, self).__setitem__(key, value)

    # def __setattr__(self, key, value):
    #     """Set the attribute using the given key and value. That is, instead of `D[key] = value`, you can do
    #     `D.key = value`. By default, this creates a tensor with shape (num_steps + 1, self.num_trajectories, 1).
    #
    #     Warnings: avoid to use this.
    #     """
    #     self.create_new_entry(key, shapes=1)
