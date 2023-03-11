#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the replay memories which stores transitions (states, actions, next_states, rewards) at each time step,
when running a policy in the environment.

It basically builds the dataset on which the various approximators will be evaluated on.
The storage can be used to train a policy, dynamic model, and/or value function approximator.

See Also:
    - `pyrobolearn.samplers`: this defines how to samples from the storages.
"""

import collections
import copy
import pickle
import queue
import numpy as np
import torch

from pyrobolearn import logger


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Storage(object):
    """Main abstract storage class."""

    @property
    def size(self):
        """Return the size of the storage. Need to be implemented in the child class."""
        return 0

    @property
    def filled_size(self):
        """Return the filled size of the storage."""
        return self.size

    @staticmethod
    def load(filename):
        """Load the storage from the disk."""
        return pickle.load(open(filename, 'r'))

    def save(self, filename):
        """Save the storage on the disk."""
        pickle.dump(self, open(filename, 'wb'))

    def get_batch(self, indices):
        """Return a batch of the storage as a `Storage` type.

        Args:
            indices (list of int): indices. Each index must be between 0 and the size of the storage.

        Returns:
            Storage: batch containing a part of the storage.
        """
        pass

    # def __repr__(self):
    #     """Return a string representing the class."""
    #     return self.__class__.__name__
    #
    # def __str__(self):
    #     """Return a string representing the class."""
    #     return self.__class__.__name__


class PyTorchStorage(Storage):
    r"""PyTorch storage.
    """

    def __init__(self, device=None, dtype=None):
        """
        Initialize the PyTorch storage.

        Args:
            device (str, torch.device, None): device to which the tensor will be allocated. If string, it can be
                'cpu' or 'cuda'. If None, it will keep the original device to which the tensor is allocated.
            dtype (torch.dtype, None): data type of a `torch.Tensor`. If None, it will keep the original dtype.
        """
        self.device = device
        self.dtype = dtype

    ##############
    # Properties #
    ##############

    @property
    def device(self):
        """Return the device instance."""
        return self._device

    @device.setter
    def device(self, device):
        """Set the device instance."""
        if device is not None:
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._device = device

    @property
    def dtype(self):
        """Return the data type."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set the data type."""
        if not (dtype is None or isinstance(dtype, torch.dtype)):
            raise ValueError("Expecting `dtype` to be None or an instance of `torch.dtype`, instead got: "
                             "{}".format(dtype))
        self._dtype = dtype

    ###########
    # Methods #
    ###########

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
        self.device, self.dtype = device, dtype
        self._to(self, device=self.device, dtype=self.dtype)

    def _to(self, item, device=None, dtype=None):
        """Send the item to the specified device and convert it to the specified data type."""
        if isinstance(item, torch.Tensor):
            logger.debug('setting tensor of size {} to {} with dtype={}'.format(item.size(), device, dtype))
            item = item.to(device=device, dtype=dtype)
        elif isinstance(item, np.ndarray):
            if item.dtype != object:  # double, float, float16, int64, int32, and uint8
                item = torch.from_numpy(item).to(device=device, dtype=dtype)
        elif isinstance(item, (float, int, np.generic)):
            item = torch.tensor(item).to(device=device, dtype=dtype)
        elif isinstance(item, dict):
            for key, value in item.items():
                item[key] = self._to(value, device=device, dtype=dtype)
        elif isinstance(item, set):
            for value in item:
                item.remove(value)
                value = self._to(value, device=device, dtype=dtype)
                item.add(value)
        elif isinstance(item, collections.Iterable):
            item = [self._to(value, device=device, dtype=dtype) for value in item]
            # for idx, value in enumerate(item):
            #     item[idx] = self._to(value, device=device, dtype=dtype)
        return item

    @staticmethod
    def cuda_is_available():
        """Returns a bool indicating if CUDA is currently available."""
        return torch.cuda.is_available()

    @staticmethod
    def cuda_device_count():
        """Returns the number of GPUs available."""
        return torch.cuda.device_count()

    def is_cuda(self, recursive_check=False):
        """Check if the tensors are on the GPU.

        Args:
            recursive_check (bool): If True, it will check recursively in the container if all the tensors are on the
                GPU. This operation might be time-consuming depending on the size of the container / storage. Normally,
                when adding new tensors they will have be sent to the correct device, so there should not be a need to
                set that variable to True. If False, it will just check the :attr:`device` member variable, and
                return True if it is 'cuda:*'. Note that if other elements are in the storage / container such as
                classes or numpy arrays, they will be discarded.
        """
        if self.device is None or recursive_check:
            def check(items):
                if isinstance(items, torch.Tensor):
                    if not items.is_cuda:
                        return False
                if isinstance(items, dict):
                    for item in items.values():
                        check(item)
                elif isinstance(items, collections.Iterable):
                    for item in items:
                        check(item)
            check(self)
            return True
        if self.device.type == 'cuda':
            return True
        return False

    def _convert_to_tensor(self, data):
        """Convert the given data to a tensor and to the correct data type if possible.

        Args:
            data (np.array, float, int, list, tuple, np.generic, torch.Tensor): data
        """
        if isinstance(data, torch.Tensor):
            return data.to(dtype=self.dtype)
        elif isinstance(data, np.ndarray):
            if data.dtype != object:  # double, float, float16, int64, int32, and uint8
                return torch.from_numpy(data).to(dtype=self.dtype)
        elif isinstance(data, (float, int, list, tuple, np.generic)):
            return torch.tensor(data).to(dtype=self.dtype)
        else:
            raise TypeError("Expecting the given data {} to be a float, int, list, tuple, np.array, np.generic, or "
                            "a torch.Tensor, instead got: {}".format(data, type(data)))


class ListStorage(list, PyTorchStorage):
    r"""PyTorch list storage

    List storage (data structure) which allocates the given tensor(s) to the specified device.
    """

    def __init__(self, args=None, device='cpu', dtype=torch.float):
        """
        Initialize the PyTorch list storage.

        Args:
            args (list, iterator): list iterator
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
        """
        PyTorchStorage.__init__(self, device=device, dtype=dtype)

        if args is None:
            args = list()
        else:
            args = list(args)

        args = self._to(args, device=self.device, dtype=self.dtype)
        super(ListStorage, self).__init__(args)

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    def insert(self, index, item):
        """Insert item before index."""
        item = self._to(item, device=self.device, dtype=self.dtype)
        super(ListStorage, self).insert(index, item)

    def append(self, item):
        """Append item to the end of the list."""
        item = self._to(item, device=self.device, dtype=self.dtype)
        super(ListStorage, self).append(item)

    def extend(self, iterable):
        """Extend list by appending elements from the iterable."""
        if not isinstance(iterable, collections.Iterable):
            raise TypeError("Expecting an iterable.")
        iterable = self._to(iterable, device=self.device, dtype=self.dtype)
        super(ListStorage, self).append(iterable)

    def __setitem__(self, key, value):
        """Set the specified value at the specified key."""
        super(ListStorage, self).__setitem__(key, self._to(value, device=self.device, dtype=self.dtype))


class FIFOQueueStorage(queue.Queue, PyTorchStorage):
    r"""FIFO Queue Storage

    FIFO queue storage (data structure) which allocates the given tensor(s) to the specified device.
    """

    def __init__(self, maxsize=0):
        """Initialize the FIFO Queue storage.

        Args:
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
        """
        super(FIFOQueueStorage, self).__init__(maxsize)

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    def put(self, item, block=False, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot
        is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else raise
        the Full exception ('timeout' is ignored in that case).
        """
        if not self.full():
            item = self._to(item, device=self.device, dtype=self.dtype)
            super(FIFOQueueStorage, self).put(item, block=block, timeout=timeout)

    def put_nowait(self, item):
        """
        Put an item into the queue without blocking. Only enqueue the item if a free slot is immediately available.
        Otherwise raise the Full exception.
        """
        item = self._to(item, device=self.device, dtype=self.dtype)
        super(FIFOQueueStorage, self).put_nowait(item)

    def __len__(self):
        """Return the size of the Queue."""
        return self.qsize()

    def __iter__(self):
        """Return the iterator object itself."""
        self.cnt = 0
        return self

    def __next__(self):  # only valid in Python 3
        """Return the next item in the sequence."""
        if self.cnt < self.qsize():
            self.cnt += 1
            return self.queue[self.cnt-1]
        else:
            raise StopIteration

    def next(self):  # for Python 2
        """Return the next item in the sequence."""
        return self.__next__()


class LIFOQueueStorage(queue.LifoQueue, PyTorchStorage):
    r"""LIFO Queue Storage

    LIFO queue storage (data structure) which allocates the given tensor(s) to the specified device.
    """

    def __init__(self, maxsize=0):
        """Initialize the LIFO Queue storage.

        Args:
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
        """
        super(LIFOQueueStorage, self).__init__(maxsize)

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    def put(self, item, block=False, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot
        is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else raise
        the Full exception ('timeout' is ignored in that case).
        """
        if not self.full():
            item = self._to(item, device=self.device, dtype=self.dtype)
            super(LIFOQueueStorage, self).put(item, block=block, timeout=timeout)

    def put_nowait(self, item):
        """
        Put an item into the queue without blocking. Only enqueue the item if a free slot is immediately available.
        Otherwise raise the Full exception.
        """
        item = self._to(item, device=self.device, dtype=self.dtype)
        super(LIFOQueueStorage, self).put_nowait(item)

    def __len__(self):
        """Return the size of the Queue."""
        return self.qsize()

    def __iter__(self):
        """Return the iterator object itself."""
        self.cnt = 0
        return self

    def __next__(self):  # only valid in Python 3
        """Return the next item in the sequence."""
        if self.cnt < self.qsize():
            self.cnt += 1
            return self.queue[self.cnt-1]
        else:
            raise StopIteration

    def next(self):  # for Python 2
        """Return the next item in the sequence."""
        return self.__next__()


class PriorityQueueStorage(queue.PriorityQueue, PyTorchStorage):
    r"""Priority Queue Storage

    Priority queue storage (data structure) which allocates the given tensor(s) to the specified device.

    Note that `queue.PriorityQueue` is a thread-safe class that use the `heapq` module (which is initially not thread
    safe) under the hood.
    """

    def __init__(self, maxsize=0, ascending=True):
        """Initialize the LIFO Queue storage.

        Args:
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
            ascending (bool): if True, the item with the lowest priority will be the first one to be retrieved.
        """
        super(PriorityQueueStorage, self).__init__(maxsize)
        self.ascending = ascending

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    def put(self, item, block=False, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot
        is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else raise
        the Full exception ('timeout' is ignored in that case).
        """
        if not self.full():
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Expecting the item to be a tuple of length 2 with (priority number, data), instead "
                                "got: {}".format(item))
            if self.ascending:
                item = (item[0], self._to(item[1], device=self.device, dtype=self.dtype))
            else:
                item = (-item[0], self._to(item[1], device=self.device, dtype=self.dtype))
            super(PriorityQueueStorage, self).put(item, block=block, timeout=timeout)

    def put_nowait(self, item):
        """
        Put an item into the queue without blocking. Only enqueue the item if a free slot is immediately available.
        Otherwise raise the Full exception.
        """
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Expecting the item to be a tuple of length 2 with (priority number, data), instead "
                            "got: {}".format(item))
        if self.ascending:
            item = (item[0], self._to(item[1], device=self.device, dtype=self.dtype))
        else:
            item = (-item[0], self._to(item[1], device=self.device, dtype=self.dtype))
        super(PriorityQueueStorage, self).put_nowait(item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until an item is
        available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises the Empty
        exception if no item was available within that time. Otherwise ('block' is false), return an item if one is
        immediately available, else raise the Empty exception ('timeout' is ignored in that case).
        """
        item = super(PriorityQueueStorage, self).get(block=block, timeout=timeout)
        if not self.ascending:
            item = (-item[0], item[1])
        return item

    def __len__(self):
        """Return the size of the Queue."""
        return self.qsize()

    def __iter__(self):
        """Return the iterator object itself."""
        self.cnt = 0
        return self

    def __next__(self):  # only valid in Python 3
        """Return the next item in the sequence."""
        if self.cnt < self.qsize():
            self.cnt += 1
            return self.queue[self.cnt-1]
        else:
            raise StopIteration

    def next(self):  # for Python 2
        """Return the next item in the sequence."""
        return self.__next__()


class SetStorage(set, PyTorchStorage):
    r"""PyTorch set storage

    Set storage (data structure) which allocates the given tensor(s) to the specified device.
    """

    def __init__(self, args=None, device='cpu', dtype=torch.float):
        """
        Initialize the PyTorch set storage.

        Args:
            args (set, iterator): list iterator
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
        """
        PyTorchStorage.__init__(self, device=device, dtype=dtype)

        if args is None:
            args = set([])
        else:
            args = set(args)

        args = self._to(args, device=self.device, dtype=self.dtype)
        super(SetStorage, self).__init__(args)

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    def add(self, item):
        """Add new item in the set."""
        item = self._to(item, device=self.device, dtype=self.dtype)
        super(SetStorage, self).add(item)


class DictStorage(dict, PyTorchStorage):
    r"""PyTorch Dict Storage

    Dictionary Storage (data structure) which allocates the given tensor to the specified device.
    The advantage of such storage class is that it is dynamic and flexible; you can decide at runtime the attributes of
    that class. So any classes can easily modify it; modify the value associated with a specific key, add or remove
    other keys with their values, etc.
    """

    def __init__(self, kwargs=None, device='cpu', dtype=torch.float, update=True):
        """
        Initialize the PyTorch dictionary storage.

        Args:
            kwargs (dict): initial dictionary.
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
            update (bool): If True, it will send the given tensors to the specified device and convert them into
                the specified data type.
        """
        PyTorchStorage.__init__(self, device=device, dtype=dtype)

        if kwargs is None:
            kwargs = dict()
        else:
            kwargs = dict(kwargs)

        if update:
            kwargs = self._to(kwargs, device=self.device, dtype=self.dtype)
        super(DictStorage, self).__init__(kwargs)

    ##############
    # Properties #
    ##############

    @property
    def size(self):
        """Return the size of the storage."""
        return len(self)

    ###########
    # Methods #
    ###########

    def update(self, dictionary, **kwargs):
        """Update the current dictionary from the other given dictionary / iterable.

        If other is a dict, does: for k in other: self[k] = other[k]
        If other is an iterable, does: for k in other: self[k] = other[k]
        """
        dictionary = self._to(dictionary, device=self.device, dtype=self.dtype)
        kwargs = self._to(kwargs, device=self.device, dtype=self.dtype)
        super(DictStorage, self).update(dictionary, **kwargs)

    def setdefault(self, key, default=None):
        """self.get(key, default), also set self[key] = default if key not in D."""
        if default is not None:
            default = self._to(default, device=self.device, dtype=self.dtype)
        super(DictStorage, self).setdefault(key, default)

    def end(self, *args, **kwargs):
        """End; fill the remaining value. This has to be inherited in the child classes."""
        pass

    #############
    # Operators #
    #############

    def __setitem__(self, key, value):
        """Add the new value in the dictionary."""
        value = self._to(value, device=self.device, dtype=self.dtype)
        super(DictStorage, self).__setitem__(key, value)

    def __getattr__(self, name):
        """Get the attribute using the key name. That is, instead of `D['name']`, you can do `D.name`."""
        return self[name]

    # def __setattr__(self, key, value):
    #     """Set the attribute using the given key and value. That is, instead of `D[key] = value`, you can do
    #     `D.key = value`"""
    #     self[key] = value


# alias
class Batch(DictStorage):
    r"""Batch storage

    The Batch storage contains states, actions, rewards, and masks. It might be filled later by the various estimators,
    targets, and returns that are defined in `pyrobolearn/returns` folder.

    The Batch is notably returned by rollout storages (used for on-policy algorithms) and experience replays (used for
    off-policy algorithms) when requesting batches.

    For the states, actions, rewards, and masks; these can be accessed from the batch using `batch[name]`. For the
    estimators, returns, targets, and others, they can be accessed using `batch[object]`.

    Finally, the Batch is created by the various storages, filled by the various returns/targets/estimators (see
    `pyrobolearn/returns` folder), and given to the various losses (see `pyrobolearn/losses` folder). Thus, there is a
    tight coupling between these 3 concepts. As for the original storages from which the batches are created, these
    are filled by the exploration phase in RL algorithms (see `pyrobolearn/algos/explorer`).
    """

    def __init__(self, kwargs=None, device=None, dtype=None, size=None, verbose=0):
        """
        Initialize the Batch storage.

        Args:
            kwargs (dict): initial dictionary containing the various states, acti
            device (torch.device, str, None): the device to put the data on (e.g. `torch.device("cuda:0")` or
               `torch.device("cpu")`). If string, it can be 'cpu' or 'cuda'. If None, it will keep the original device
               to which the tensor is allocated.
            dtype (torch.dtype, None): convert the `torch.Tensor` to the specified data type. If None, it will keep
                the original dtype
            size (int, None): size of the batch.
            verbose (int, bool): if True, it will print information about the batch status, such as what is being
                inserted, removed, etc. Don't use it on states or actions that are big.
        """
        if verbose == 1:
            print("\nCreating batch...")

        super(Batch, self).__init__(kwargs=kwargs, device=device, dtype=dtype, update=False)
        # contains the current values evaluated during the update phase of RL algorithms.
        self.current = DictStorage(kwargs={}, device=device, dtype=dtype, update=False)
        # indices where the masks is different from 0 in the current batch
        self.indices = None
        self._size = size if size is not None else len(kwargs['masks'])   # TODO: need to generalize this

        if verbose > 1:
            print("\nCreating batch with the following variables: ")

            if 'states' in kwargs:
                states = kwargs['states'][0]  # only take the first state for now
                print("states: {}".format(torch.cat((torch.arange(len(states),
                                                                  dtype=torch.float).view(-1, 1),
                                                     states), dim=1)))

            if 'actions' in kwargs:
                actions = kwargs['actions'][0]  # only take the first action for now
                print("actions: {}".format(torch.cat((torch.arange(len(actions),
                                                                   dtype=torch.float).view(-1, 1),
                                                      actions), dim=1)))

            if 'rewards' in kwargs:
                rewards = kwargs['rewards']
                print("rewards: {}".format(torch.cat((torch.arange(len(rewards), dtype=torch.float).view(-1, 1),
                                                      rewards), dim=1)))

            if 'masks' in kwargs:
                masks = kwargs['masks']
                print("masks: {}".format(torch.cat((torch.arange(len(masks), dtype=torch.float).view(-1, 1),
                                                    masks), dim=1)))

            if 'returns' in kwargs:
                returns =kwargs['returns']
                print("returns: {}".format(torch.cat((torch.arange(len(returns), dtype=torch.float).view(-1, 1),
                                                      returns), dim=1)))

            # print other variables
            tmp = {'states', 'actions', 'rewards', 'masks', 'returns'}
            for key, value in kwargs.items():
                if key not in tmp:
                    print("{}: {}".format(key, value))

        if verbose:
            print("Batch created.")

    @property
    def size(self):
        """Return the size of the batch."""
        return self._size

    def get_current(self, key, default=None):
        """Try first to get the key from :attr:`current`, if not present, try to get it from the batch storage.

        Args:
            key (object): dictionary key
            default (object): default value to return if the key is not found in `Batch.current` and `Batch`.
        """
        # check in current
        if key in self.current:
            return self.current[key]
        # check in self
        if key in self:
            return self[key]
        # return default
        return default

    # def __contains__(self, key):
    #     """Check if the given key is in batch.current and in batch."""
    #     return (key in self.current) or (key in self)


class RolloutStorage(DictStorage):  # TODO: think about when multiple policies: storage[policy_class]['actions']?
    r"""Rollout Storage

    Specific storage used in RL which stores transitions at each step `(s_t, a_t, s_{t+1}, r_t)`, and allows to
    also store other values such as the log probability of policy distribution :math:`\log \pi_{\theta}(a|s)`,
    value scalar returned by value functions (such as :math:`V_{\phi}(s_t)`, :math:`Q_{\phi}(s_t, a_t)`,
    :math:`A_{\phi}(s_t, a_t)`, etc), return / estimator values (such as :math:`R(\tau) = \sum_{t=0}^T \gamma^t r_t`,
    and others).

    The code here was inspired by [1] but modified such that it is more dynamic and flexible. This is achieved by
    inheriting from the PyTorch `DictStorage` class which allows you to use the storage as if it was a dictionary but
    allows you to send the tensors to the specified device. Note that these flexibility and dynamic properties can
    render the code more prone to runtime errors, so use it with extra care! For instance, before trying to access a
    key variable check that it is correctly present inside the storage. Nonetheless, having a dynamic rollout storage
    has its advantages, as you can for example store multiple value scalars from multiple value function approximators.

    Also, in contrast to [1], we do not compute the returns / estimators here. This is done by the `Estimator` class
    which takes as input a `RolloutStorage`, and will insert them inside the storage.

    In PRL, this storage is notably used by `RLAlgo` (`Explorator`, `Evaluator`, `Updater`), `Loss`, `Estimators`, etc.

    References:
        - [1] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
    """

    def __init__(self, num_steps, state_shapes, action_shapes, num_trajectories=1, verbose=0):
        # , recurrent_hidden_state_size=0):
        """
        Initialize the rollout storage.

        Args:
            num_steps (int): number of steps in one episode
            state_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an observation/state.
            action_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an action.
            num_trajectories (int): number of trajectories.
            verbose (int, bool): verbose level, if False (=0) it won't print anything. If True (=1) it will print basic
                information. If verbose=2, it will print information about what is being inserted and removed in the
                storage, and the batch status.
        """
        # recurrent_hidden_state_size (int): size of the internal state
        super(RolloutStorage, self).__init__()
        self._step = np.zeros(int(num_trajectories), dtype=np.int)
        self._num_steps = int(num_steps)
        self._num_trajectories = int(num_trajectories)
        self._state_shapes = state_shapes
        self._action_shapes = action_shapes
        self._shifts = {}  # dictionary that maps the key to the time shift; this is add to the current time step
        self.verbose = verbose

        if self.verbose:
            print("\nCreating RolloutStorage with num_steps={} and num_rollouts={}".format(self._num_steps,
                                                                                           self._num_trajectories))

        self.init(self.num_steps, state_shapes, action_shapes, self.num_trajectories)

        if self.verbose:
            print("RolloutStorage created.")

    ##############
    # Properties #
    ##############

    @property
    def num_steps(self):
        """Return the number of time steps (used in the finite-horizon RL setting)."""
        return self._num_steps

    @property
    def num_trajectories(self):
        """Return the number of processes used."""
        return self._num_trajectories

    @property
    def size(self):
        """Return the size (=number of steps * number of processes) of the rollout storage."""
        return self._num_steps * self._num_trajectories

    @property
    def filled_size(self):
        """Return the filled size by looking at the entries where the mask = 1."""
        # print("Mask: {}".format(self.masks[:, :, 0]))
        # print("Mask shape: {}".format(self.masks.shape))
        # print("Steps: {}".format(self._step))
        return np.minimum(len(self.masks[self.masks == 1]), self.size)

    @property
    def capacity(self):
        """Return the capacity of the rollout storage (=number of steps * number of processes)."""
        return self._num_steps * self._num_trajectories

    @property
    def curr_step(self):
        """Return the current time step."""
        return self._step

    @property
    def state_shapes(self):
        """Return the state shapes."""
        return self._state_shapes

    @property
    def action_shapes(self):
        """Return the action shapes."""
        return self._action_shapes

    ###########
    # Methods #
    ###########

    def step(self, rollout_idx=0):
        """Perform one step; increment by one the current step. If it reaches the end, start from 0 again.

        Args:
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
        """
        # if end of storage, go at the beginning
        self._step[rollout_idx] = (self._step[rollout_idx] + 1) % (self.num_steps + 1)  # self.num_steps

    def create_new_entry(self, key, shapes, num_steps=None, dtype=torch.dtype):
        """Create a new entry (=tensor) in the rollout storage dictionary. The tensor will have the dimension
        (num_steps, self.num_trajectories, *shape) for each shape in shapes, and will be initialized to zero.
        The tensor will also have the same type than the other tensors and will be sent to the correct device.

        Args:
            key (str, object): key of the dictionary.
            shapes (list of tuple of int, tuple of int, int): (list of) shape(s) of the tensor(s).
            num_steps (int, None): the number of time steps. This value can not be smaller than `self.num_steps`.
                If None, `self.num_steps` will be used.
            dtype (torch.dtype, np.generic, object): specify if we want to allocate a `torch.Tensor`, or a numpy array.
                If dtype == torch.dtype, then it will be set to dtype = self.dtype.
        """
        # check the key
        # if not isinstance(key, str):
        #     raise TypeError("Expecting the key to be a string, instead got: {} with type {}".format(key, type(key)))

        # check the number of time steps
        if num_steps is None:
            num_steps = self.num_steps
        if num_steps < self.num_steps:
            raise ValueError("Expecting the given number of time steps (={}) to be bigger than the initialized number "
                             "of time steps (={})".format(num_steps, self.num_steps))

        # allocate the new tensor

        # convert torch type if necessary
        if dtype == torch.dtype:
            dtype = self.dtype

        # if we have a list of shapes
        if isinstance(shapes, list):
            if isinstance(dtype, torch.dtype):
                self[key] = [torch.zeros(num_steps, self.num_trajectories, *shape).to(device=self.device, dtype=dtype)
                             for shape in shapes]
            else:  # numpy array
                self[key] = [np.zeros((num_steps, self.num_trajectories,) + shape, dtype=dtype) for shape in shapes]

        # if the 'shapes' is a tuple
        elif isinstance(shapes, tuple):
            if isinstance(dtype, torch.dtype):
                self[key] = torch.zeros(num_steps, self.num_trajectories, *shapes).to(device=self.device, dtype=dtype)
            else:
                self[key] = np.zeros((num_steps, self.num_trajectories,) + shapes, dtype=dtype)

        # if the 'shapes' is an int
        elif isinstance(shapes, int):
            if isinstance(dtype, torch.dtype):
                self[key] = torch.zeros(num_steps, self.num_trajectories, shapes).to(device=self.device, dtype=dtype)
            else:
                self[key] = np.zeros((num_steps, self.num_trajectories, shapes), dtype=dtype)

        else:
            raise TypeError("Expecting the given shapes {} to be a list of tuple of int, a tuple of int, or an int, "
                            "instead got: {}".format({}, type(shapes)))

        if self.verbose:
            print("Storage: creating new entry for {} with shapes {}".format(key, shapes))

        # add shift
        self._shifts[key] = num_steps - self.num_steps

    def init(self, num_steps, state_shapes, action_shapes, num_trajectories=1):
        """
        Initialize the rollout storage by allocating the appropriate tensors for the observations (states), actions,
        rewards, masks, and returns.

        Note that some values go from 0 to T (where T is the finite-time horizon), while others go from 0 to T+1.

        Args:
            num_steps (int): number of time steps in the finite-horizon RL setting.
            state_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an observation/state.
            action_shapes (list of tuple of int, tuple of int): each tuple represents the shape of an action.
            num_trajectories (int): number of trajectories.
        """
        # clear itself: remove all items from the DictStorage, and reset all variables
        self.clear()
        self._step = np.zeros(int(num_trajectories), dtype=np.int)
        self._num_steps = int(num_steps)
        self._num_trajectories = int(num_trajectories)
        self._state_shapes = state_shapes
        self._action_shapes = action_shapes

        # allocate space for observations / states
        logger.debug('creating space for states with shape: {}'.format(state_shapes))
        if not isinstance(state_shapes, list):
            state_shapes = [state_shapes]
        self.create_new_entry('states', shapes=state_shapes, num_steps=self.num_steps + 1)

        # allocate space for actions
        logger.debug('creating space for actions with shape: {}'.format(action_shapes))
        if not isinstance(action_shapes, list):
            action_shapes = [action_shapes]
        self.create_new_entry('actions', shapes=action_shapes, num_steps=self.num_steps)

        # allocate space for rewards
        logger.debug('creating space for rewards')
        self.create_new_entry('rewards', shapes=1, num_steps=self.num_steps)

        # allocate space for the masks
        logger.debug('creating space for masks')
        self.create_new_entry('masks', shapes=1, num_steps=self.num_steps + 1)

        # allocate space for action distribution
        self.create_new_entry('action_distributions', shapes=[() for _ in action_shapes], num_steps=self.num_steps,
                              dtype=object)

        # space for log probabilities on policy, distributions, scalar values from value functions,
        # recurrent hidden states, and others have to be allocated outside the class

    def reset(self, init_states=None, rollout_idx=0, *args, **kwargs):
        """Reset the storage by copying the last value and setting it to the first value.

        Args:
            init_states (torch.Tensor, list of torch.Tensor): (list of) initial state(s) / observation(s).
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
        """
        # reset the step
        self._step[rollout_idx] = 0

        # reset masks
        # self.masks[:, rollout_idx].copy_(torch.ones_like(self.masks[:, rollout_idx]))  # fill with ones
        self.masks[:, rollout_idx].copy_(torch.zeros_like(self.masks[:, rollout_idx]))  # fill with zeros

        # insert initial states
        if init_states is None:
            for state in self.states:
                state[0, rollout_idx].copy_(state[-1, rollout_idx])
        else:
            if not isinstance(init_states, list):
                init_states = [init_states]
            for observation, value in zip(self.states, init_states):
                observation[0, rollout_idx].copy_(self._convert_to_tensor(value))
        # self.masks[0, rollout_idx].copy_(self.masks[-1, rollout_idx])
        # self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])

    def update_tensor(self, key, values, step=None, rollout_idx=None, copy=True):
        """
        Update one particular (or several) tensor(s) in the dictionary at the specified time step. It will convert the
        given value tensors to the correct data type if not already done.

        Args:
            key (object): dictionary key
            values ((list of) torch.Tensor, np.array, int, float, np.generic): items
            step (None, int): the time step.
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
            copy (bool): if the item(s) should be copied. If False, it will not copy the item(s). Note that if you
                modify these item(s) outside the storage, it will be reflected in the storage as well.
        """
        # check the given time step
        if step is None:
            step = self._step[rollout_idx]

        # check if the key is inside the storage
        if key in self:

            # perform: tensor[step] = value
            def set_tensor(tensor, step, value, copy):
                # if torch tensor
                if isinstance(tensor, torch.Tensor):
                    if copy:
                        tensor[step][rollout_idx].copy_(self._convert_to_tensor(value))
                    else:
                        tensor[step][rollout_idx] = self._convert_to_tensor(value)

                # if numpy array
                elif isinstance(tensor, np.ndarray):
                    if copy:
                        tensor[step][rollout_idx] = np.copy(value)
                    else:
                        tensor[step][rollout_idx] = value

            # if we have a list of tensors at the specified key
            if isinstance(self[key], list):

                # if we are given a list of tensors, we update each tensor
                if isinstance(values, list):
                    for tensor, value in zip(self[key], values):
                        set_tensor(tensor, step, value, copy=copy)

                # if we are given one value, we put it to each tensor
                else:
                    for tensor in self[key]:
                        set_tensor(tensor, step, values, copy=copy)

            # if we have a torch.tensor or numpy array
            else:
                set_tensor(self[key], step, values, copy=copy)

    def insert(self, states, actions, next_states, reward, mask, distributions=None, update_step=True, rollout_idx=0,
               **kwargs):
        # distributions, values=None):
        # recurrent_hidden_state, action_log_prob):
        """
        Insert the given parameters into the storage.

        Args:
            states (torch.Tensor, list of torch.Tensor): (list of) state(s) / observation(s).
            actions (torch.Tensor, list of torch.Tensor): (list of) action(s).
            next_states (torch.Tensor, list of torch.Tensor): (list of) next state(s) / observation(s).
            reward (float, int, torch.Tensor): reward value
            mask (float, int, torch.Tensor): masks. They are set to zeros after an episode has terminated.
            distributions (torch.distributions.Distribution, None): action distribution.
            update_step (bool): if True, it will update the current time step. If False, the user needs to call
                `step()` in order to update it.
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
            **kwargs (dict): dictionary containing other parameters to update in the storage. The other parameters
                had to be added using the `create_new_entry()` method.
        """
        t = self._step[rollout_idx]

        if self.verbose > 1:
            print("\nStorage: rollout = {}, step = {}".format(rollout_idx, t))
            print("Storage: insert state: {}".format(states))
            print("Storage: insert action: {}".format(actions))
            print("Storage: insert next state: {}".format(next_states))
            print("Storage: insert reward: {}".format(reward))
            print("Storage: insert mask: {}".format(mask))

        # check given observations/states and actions
        if not isinstance(next_states, list):
            next_states = [next_states]
        if not isinstance(actions, list):
            actions = [actions]
        if not isinstance(distributions, list):
            distributions = [distributions]

        # insert each observation / action
        for observation, storage in zip(next_states, self['states']):
            storage[t + 1][rollout_idx].copy_(self._convert_to_tensor(observation))
        for action, storage in zip(actions, self['actions']):
            storage[t][rollout_idx].copy_(self._convert_to_tensor(action))

        # insert rewards and masks
        self['rewards'][t][rollout_idx].copy_(self._convert_to_tensor(reward))
        if mask is None:
            mask = torch.tensor(1.)
        self['masks'][t + 1][rollout_idx].copy_(self._convert_to_tensor(mask))

        # insert distributions
        for distribution, storage in zip(distributions, self['action_distributions']):
            storage[t][rollout_idx] = distribution

        # add other elements
        for key, value in kwargs:
            if key in self and key in self._shifts:
                self.update_tensor(key, value, step=self._step+self._shifts[key], rollout_idx=rollout_idx, copy=True)
            else:
                raise ValueError("The given keys in kwargs do not exist in this storage or in its 'shift' dictionary.")

        # update the step if specified
        if update_step:
            self.step(rollout_idx=rollout_idx)

    def add_trajectory(self, trajectory, rollout_idx=0):
        r"""
        Add a trajectory/rollout [(s_t, a_t, s_{t+1}, r_t, d_t)]_{t=1}^T in the storage. This calls in for-loop the
        `insert` method.

        Args:
            trajectory (list of dict): trajectory represented as a list of dictionaries where each dictionary contains
                a transition tuple (s_t, a_t, s_{t+1}, r_t, d_t), and thus has at least the following key: `states`,
                `actions`, `next_states`, `reward`, `mask`.
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
        """
        # insert each step in the trajectory into the storage
        for step in trajectory:
            self.insert(rollout_idx=rollout_idx, **step)

        # fill remaining mask values to be 0 (because the episode is done)
        self.end(rollout_idx=rollout_idx)

    def get_batch(self, indices):
        """Return a batch of the Rollout storage in the form of a `DictStorage`.

        Args:
            indices (list of int): indices. Each index must be between 0 and `self.filled_size`-1.

        Returns:
            DictStorage / Batch: batch containing a part of the storage. Variables such as `states`, `actions`,
                `rewards`, `masks`, and others can be accessed from the object.
        """
        # In the next comments, T = number of time steps, P = number of processes, and I = number of indices, F =
        # number of filled entries (where the mask == 1)
        batch = {}
        size = len(indices)

        # # The following sampling method was not optimal as it would sample entries where the mask == 0 (i.e. when
        # # the episode is over)
        # def sample(item, indices):
        #     if isinstance(item, torch.Tensor):
        #         if len(item) == self.num_steps + 1:  # = T+1
        #             item = item[:-1]  # take only the T steps
        #         return item.view(-1, *item.size()[2:])[indices]  # reshape to (T*P, *shape) and from T*P takes I
        #
        #     elif isinstance(item, np.ndarray):
        #         if len(item) == self.num_steps + 1:  # = T+1
        #             item = item[:-1]  # take only the T steps
        #         return item.reshape(-1, *item.shape[2:])[indices]  # reshape to (T*P, *shape) and from T*P takes I

        if self.verbose:
            print("\nStorage: get batch with size: {} and indices: {}".format(len(indices), indices))

        def sample(item, indices):
            """Given indices where each index is between 0 and `self.filled_size` (=number of masks that are equal
            to 1), it returns the corresponding entries in the item.
            """
            if isinstance(item, (torch.Tensor, np.ndarray)):
                return item[indices[:, 0], indices[:, 1]]  # [I, *shape]
            else:
                raise TypeError("Expecting the given 'item' to be a torch.Tensor or np.ndarray, but got: "
                                "{}".format(type(item)))

        # compute the step and traj indices
        original_indices = indices
        masks = (self.masks[:, :, 0] == 1).nonzero()  # [F, 2] --> indices for (step_idx, traj_idx)
        masks[:, 0] -= 1  # remove 1 because indices for masks are in [1, t+1] --> [0, t]
        indices = masks[indices]  # [I,2] --> allowed indices for (step_idx, traj_idx)

        # if self.verbose:
        #     print("Mask length: {}".format(len(masks)))
        #     print("Indices: {}".format(indices))

        # go through each attribute and sample from the tensors
        for key, value in self.items():
            # print("batch - add key: {}".format(key))
            if isinstance(value, list):  # value = list of tensors
                batch[key] = [sample(val, indices) for val in value]  # [[I, *shape] for each shape]
            else:  # value = tensor
                # if key == 'masks':  # TODO: need to shift masks?
                #     m = torch.clone(masks)
                #     m[:, 0] += 1  # indices [0, t] --> [1, t+1]
                #     idx = m[original_indices]
                #     batch[key] = value[idx[:, 0], idx[:, 1]]
                # else:
                batch[key] = sample(value, indices)  # [I, *shape]

        # TODO: add the following lines in the `end` method? Need to be called only one time as long we don't clear
        #  replace the zeros in the action_distributions field by dummy distributions
        # take the first distribution for each action distribution
        dists = [dist[dist != 0][0] for dist in self['action_distributions']]
        # replace the zeros by that first distribution
        for distribution, dist in zip(batch['action_distributions'], dists):
            distribution[distribution == 0] = dist

        # Now that we have a list of distributions for each action, transform it to a single distribution
        distributions = batch['action_distributions']
        for i, distribution in enumerate(distributions):
            # select first distribution
            dist = distribution[0]
            # transform list of distributions to a single distribution and put it in the batch
            distributions[i] = dist.__class__.from_list(distribution)

        # create Batch object
        batch = Batch(batch, device=self.device, dtype=self.dtype, size=size, verbose=self.verbose)
        batch.indices = torch.tensor(range(len(indices)))[batch['masks'][:, 0] != 0].tolist()
        # print("Batch indices: {}".format(batch.indices))  # TODO: need to shift masks?

        # return batch (which is given to the updater (and loss))
        return batch

    def end(self, rollout_idx=0, *args, **kwargs):
        """Once arrived at the end of an episode, it will fill the remaining mask values.

        Args:
            rollout_idx (int, torch.tensor, np.array, list): trajectory/rollout index(ices). This index must be below
                `self.num_trajectories`.
        """
        t = self._step[rollout_idx]
        self['masks'][t+1:, rollout_idx] = torch.zeros(self.num_steps - t, 1)

    #############
    # Operators #
    #############

    def __setitem__(self, key, value):
        """Add the new value in the dictionary. Key must be strings or objects."""
        # if not isinstance(key, str):
        #     raise TypeError("The rollout storage only accepts key as strings! Instead got: {} with type "
        #                     "{}".format(key, type(key)))
        super(RolloutStorage, self).__setitem__(key, value)

    # def __setattr__(self, key, value):
    #     """Set the attribute using the given key and value. That is, instead of `D[key] = value`, you can do
    #     `D.key = value`. By default, this creates a tensor with shape (num_steps + 1, self.num_trajectories, 1).
    #
    #     Warnings: avoid to use this.
    #     """
    #     self.create_new_entry(key, shapes=1, num_steps=self.num_steps + 1)


# Tests
if __name__ == '__main__':
    a = DictStorage([(0, torch.tensor(range(9, 12))),
                     (1, [torch.tensor(range(3)), torch.tensor(range(3, 6))])], device='cpu')
    # a[0] = torch.tensor(range(9, 12))
    # a[1] = [torch.tensor(range(3)), torch.tensor(range(3, 6))]
    # a.to('cpu')
    print("Cuda: {}".format(a.is_cuda()))
    print(a)
    print(a[0].is_cuda)
