#!/usr/bin/env python
"""Define the State class.

This file defines the `State` class, which is returned by the environment, and given as an input to several
models such as policies/controllers, dynamic transition functions, value approximators, reward/cost function, and so on.
"""

import copy
import collections
# from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import gym

from pyrobolearn.utils.data_structures.orderedset import OrderedSet
from pyrobolearn.utils.data_structures.queues import FIFOQueue


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class State(object):
    r"""State class.

    The `State` is returned by the environment and given to the policy. The state might include information
    about the state of one or several objects in the world, including robots.

    It is the main bridge between the robots/objects in the environment and the policy. Specifically, it is given
    as an input to the policy which knows how to feed the state to the learning model. Usually, the user only has to
    instantiate a child of this class, and give it to the policy and environment, and that's it.
    In addition to the policy, the state can be given to a controller, dynamic model, value estimator, reward function,
    and so on.

    To allow our framework to be modular, we favor composition over inheritance [1] leading the state to be decoupled
    from notions such as the environment, policy, rewards, etc. This class also describes the `state_space` which has
    initially been defined in `gym.Env` [2].

    Note that the policy does not represent in a strict sense the robot but more its brain, the sensors and actuators
    are parts of the environments. Note also that any kind of data can be represented with numbers (e.g. binary code).

    Example:

        sim = Bullet()
        robot = Robot(sim)

        # Two ways to initialize states
        states = State([JntPositionState(robot), JntVelocityState(robot)])
        # or
        states = JntPositionState(robot) + JntVelocityState(robot)

        actions = JntPositionAction(robot)

        policy = NNPolicy(states, actions)

    References:
        [1] "Wikipedia: Composition over Inheritance", https://en.wikipedia.org/wiki/Composition_over_inheritance
        [2] "OpenAI gym": https://gym.openai.com/   and    https://github.com/openai/gym

    """

    def __init__(self, states=(), data=None, space=None, window_size=1, axis=None, ticks=1, name=None):
        """
        Initialize the state. The state contains some kind of data, or is a state combined of other states.

        Args:
            states (list/tuple of State): list of states to be combined together (if given, we can not specified data)
            data (np.array): data associated to this state
            space (gym.space): space associated with the given data
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
            name (str, None): name of the state. If None, by default, it will have the name of the class.

        Warning:
            Both arguments can not be provided to the state.
        """
        # Check arguments
        if states is None:
            states = tuple()

        # check that the given `states` is a list of states
        if not isinstance(states, (list, tuple, set, OrderedSet)):
            # TODO: should check that states is a list of state, however O(N)
            if data is None and isinstance(states, np.ndarray):  # this is in the case someone calls `State(data)`
                data = states
                states = tuple()
            else:
                raise TypeError("Expecting a list, tuple, or (ordered) set of states.")

        # check that the list of states and the data are not provided together
        if len(states) > 0 and data is not None:
            raise ValueError("Please specify only one of the argument `states` xor `data`, but not both.")

        # Check if the data is given, and convert it to a numpy array if necessary
        if data is not None:
            if not isinstance(data, np.ndarray):
                if isinstance(data, (list, tuple)):
                    data = np.array(data)
                elif isinstance(data, (int, float)):
                    data = np.array([data])
                else:
                    raise TypeError("Expecting a numpy array, a list/tuple of int/float, or an int/float for 'data'")

        # The following attributes should normally be set in the child classes
        self._data = data
        self._torch_data = data if data is None else torch.from_numpy(data).float()
        self._space = space
        self._distribution = None  # for sampling
        self._normalizer = None
        self._noiser = None  # for noise
        self.name = name
        self._training_mode = False

        # create ordered set which is useful if this state is a combination of multiple states
        self._states = OrderedSet()
        if self._data is None:
            self.add(states)

        # set data windows
        self._window, self._torch_window = None, None
        self.window_size = window_size  # this initializes the windows (FIFO queues)
        self.axis = axis

        # set ticks and counter
        self._cnt = 0
        self.ticks = ticks

        # reset state
        self.reset()

    ##############################
    # Properties (Getter/Setter) #
    ##############################

    @property
    def states(self):
        """
        Get the list of states.
        """
        return self._states

    @states.setter
    def states(self, states):
        """
        Set the list of states.
        """
        if self.has_data():
            raise AttributeError("Trying to add internal states to the current state while it already has some data. "
                                 "A state should be a combination of states or should contain some kind of data, "
                                 "but not both.")
        if isinstance(states, collections.Iterable):
            for state in states:
                if not isinstance(state, State):
                    raise TypeError("One of the given states is not an instance of State.")
                self.add(state)
        else:
            raise TypeError("Expecting an iterator (e.g. list, tuple, OrderedSet, set,...) over states")

    @property
    def data(self):
        """
        Get the data associated to this particular state, or the combined data associated to each state.

        Returns:
            list of np.ndarray: list of data associated to the state
        """
        # if the current state has data
        if self.has_data():
            if len(self.window) == 1:
                return [self.window[0]]  # [self._data]

            # concatenate the data in the window
            if self.axis is None:
                return [np.concatenate(self.window.queue)]

            # stack the data in the window
            return [np.stack(self.window.queue, axis=self.axis)]  # stack

        # if multiple states, return the combined data associated to each state
        return [state.data[0] for state in self._states]

    @data.setter
    def data(self, data):
        """
        Set the data associated to this particular state, or the combined data associated to each state.
        Each data will be clipped if outside the range/bounds of the corresponding state.

        Args:
            data: the data to set
        """
        if self.has_states():  # combined states
            if not isinstance(data, collections.Iterable):
                raise TypeError("data is not an iterator")
            if len(self._states) != len(data):
                raise ValueError("The number of states is different from the number of data segments")
            for state, d in zip(self._states, data):
                state.data = d

        # one state: change the data
        # if self.has_data():
        else:
            # make sure data is a numpy array
            if not isinstance(data, np.ndarray):
                if isinstance(data, (list, tuple)):
                    data = np.array(data)
                    if len(data) == 1 and self._data.shape != data.shape:  # TODO: check this line
                        data = data[0]
                elif isinstance(data, (int, float)):
                    data = data * np.ones(self._data.shape)
                else:
                    raise TypeError("Expecting a numpy array, a list/tuple of int/float, or an int/float for 'data'")

            # if previous data shape is different from the current one
            if self._data is not None and self._data.shape != data.shape:
                raise ValueError("The given data does not have the same shape as previously.")

            # clip the value using the space
            if self.has_space():
                if self.is_continuous():  # continuous case
                    low, high = self._space.low, self._space.high
                    data = np.clip(data, low, high)
                else:  # discrete case
                    n = self._space.n
                    if data.size == 1:
                        data = np.clip(data, 0, n)

            # set data
            self._data = data
            self._torch_data = torch.from_numpy(data).float()
            self.window.append(self._data)
            self.torch_window.append(self._torch_data)

            # check that the window is full: if not, copy the last data
            if len(self.window) != self.window.maxsize:
                for _ in range(self.window.maxsize - len(self.window)):
                    # copy last appended data
                    self.window.append(self.window[-1])
                    self.torch_window.append(self.torch_window[-1])

    @property
    def merged_data(self):
        """
        Return the merged data.
        """
        # fuse the data
        fused_state = self.fuse()
        # return the data
        return fused_state.data

    @property
    def last_data(self):
        """Return the last provided data."""
        if self.has_data():
            return self.window[-1]
        return [state.last_data for state in self._states]

    @property
    def torch_data(self):
        """
        Return the data as a list of torch tensors.
        """
        if self.has_data():
            if len(self.torch_window) == 1:
                return [self.torch_window[0]]  # [self._torch_data]

            # concatenate the data in the window
            if self.axis is None:
                return [torch.cat(self.torch_window.tolist())]

            # stack the data in the window
            return [torch.stack(self.torch_window.tolist(), dim=self.axis)]  # stack

        return [state.torch_data[0] for state in self._states]

    @torch_data.setter
    def torch_data(self, data):
        """
        Set the torch data and update the numpy version of the data.

        Args:
            data (torch.Tensor, list of torch.Tensors): data to set.
        """
        if self.has_states():  # combined states
            if not isinstance(data, collections.Iterable):
                raise TypeError("data is not an iterator")
            if len(self._states) != len(data):
                raise ValueError("The number of states is different from the number of data segments")
            for state, d in zip(self._states, data):
                state.torch_data = d

        # one state: change the data
        # if self.has_data():
        else:
            if isinstance(data, torch.Tensor):
                data = data.float()
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, (list, tuple)):
                data = torch.from_numpy(np.array(data)).float()
            elif isinstance(data, (int, float)):
                data = data * torch.ones(self._data.shape)
            else:
                raise TypeError("Expecting a Torch tensor, numpy array, a list/tuple of int/float, or an int/float for"
                                " 'data'")
            if self._torch_data.shape != data.shape:
                raise ValueError("The given data does not have the same shape as previously.")

            # clip the value using the space
            if self.has_space():
                if self.is_continuous():  # continuous case
                    low, high = torch.from_numpy(self._space.low), torch.from_numpy(self._space.high)
                    data = torch.min(torch.max(data, low), high)
                else:  # discrete case
                    n = self._space.n
                    if data.size == 1:
                        data = torch.clamp(data, min=0, max=n)

            # set data
            self._torch_data = data
            self._data = data.detach().numpy() if data.requires_grad else data.numpy()
            self.torch_window.append(self._torch_data)
            self.window.append(self._data)

            # check that the window is full: if not, copy the last data
            if len(self.window) != self.window.maxsize:
                for _ in range(self.window.maxsize - len(self.window)):
                    # copy last appended data
                    self.window.append(self.window[-1])
                    self.torch_window.append(self.torch_window[-1])

    @property
    def merged_torch_data(self):
        """
        Return the merged torch data.

        Returns:
            list of torch.Tensor: list of data torch tensors.
        """
        # fuse the data
        fused_state = self.fuse()
        # return the data
        return fused_state.torch_data

    @property
    def last_torch_data(self):
        """Return the last provided torch data."""
        if self.has_data():
            return self.torch_window[-1]
        return [state.last_torch_data for state in self._states]

    @property
    def vec_data(self):
        """
        Return a vectorized form of the data.

        Returns:
            np.array[N]: all the data.
        """
        return np.concatenate([data.reshape(-1) for data in self.merged_data])

    @property
    def vec_torch_data(self):
        """
        Return a vectorized form of all the torch tensors.

        Returns:
            torch.Tensor([N]): all the torch tensors reshaped such that they are unidimensional.
        """
        return torch.cat([data.reshape(-1) for data in self.merged_torch_data])

    @property
    def space(self):
        """
        Get the corresponding space.
        """
        if self.has_space():
            return [self._space]
        return [state._space for state in self._states]

    @space.setter
    def space(self, space):
        """
        Set the corresponding space. This can only be used one time!
        """
        if self.has_data() and not self.has_space() and isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):
            self._space = space

    @property
    def name(self):
        """
        Return the name of the state.
        """
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @name.setter
    def name(self, name):
        """
        Set the name of the state.
        """
        if name is None:
            name = self.__class__.__name__
        if not isinstance(name, str):
            raise TypeError("Expecting the name to be a string.")
        self._name = name

    @property
    def shape(self):
        """
        Return the shape of each state. Some states, such as camera states have more than 1 dimension.
        """
        return [data.shape for data in self.data]

    @property
    def merged_shape(self):
        """
        Return the shape of each merged state.
        """
        return [data.shape for data in self.merged_data]

    @property
    def size(self):
        """
        Return the size of each state.
        """
        return [data.size for data in self.data]

    @property
    def merged_size(self):
        """
        Return the size of each merged state.
        """
        return [data.size for data in self.merged_data]

    @property
    def dimension(self):
        """
        Return the dimension (length of shape) of each state.
        """
        return [len(data.shape) for data in self.data]

    @property
    def merged_dimension(self):
        """
        Return the dimension (length of shape) of each merged state.
        """
        return [len(data.shape) for data in self.merged_data]

    @property
    def num_dimensions(self):
        """
        Return the number of different dimensions (length of shape).
        """
        return len(np.unique(self.dimension))

    # @property
    # def distribution(self):
    #     """
    #     Get the current distribution used when sampling the state
    #     """
    #     pass
    #
    # @distribution.setter
    # def distribution(self, distribution):
    #     """
    #     Set the distribution to the state.
    #     """
    #     # check if distribution is discrete/continuous
    #     pass

    @property
    def in_training_mode(self):
        """Return True if we are in training mode."""
        return self._training_mode

    @property
    def window(self):
        """Return the window."""
        return self._window

    @property
    def torch_window(self):
        """Return the torch window."""
        return self._torch_window

    @property
    def window_size(self):
        """Return the window size."""
        return self.window.maxsize

    @window_size.setter
    def window_size(self, size):
        """Set the window size."""
        if size is None:
            size = 1
        if not isinstance(size, int):
            raise TypeError("Expecting the given window size to be an int, instead got: {}".format(type(size)))
        size = size if size > 0 else 1

        # create windows
        self._window = FIFOQueue(maxsize=size)
        self._torch_window = FIFOQueue(maxsize=size)

        # add data if present
        if self._data is not None:
            self._window.append(self._data)
            self._torch_window.append(self._torch_data)

            # check that the window is full: if not, copy the last data
            if len(self._window) != self._window.maxsize:
                for _ in range(self._window.maxsize - len(self._window)):
                    # copy last appended data
                    self._window.append(self._window[-1])
                    self._torch_window.append(self._torch_window[-1])

    @property
    def axis(self):
        """Return the axis to concatenate or stack the states in the current window."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        """Set the axis to concatenate or stack the states in the current window."""
        if axis is not None and not isinstance(axis, int):
            raise TypeError("Expecting the given axis to be None (concatenate) or an int (stack), instead got: "
                            "{}".format(type(axis)))
        self._axis = axis

    @property
    def ticks(self):
        """Return the number of ticks to sleep before getting the next state data."""
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        """Set the number of ticks to sleep before getting the next state data."""
        ticks = int(ticks)
        if ticks < 1:
            ticks = 1
        self._ticks = ticks

    ###########
    # Methods #
    ###########

    def train(self):
        """Set the state in training mode."""
        self._training_mode = True

    def eval(self):
        """Set the state in evaluation / test mode."""
        self._training_mode = False

    def is_combined_states(self):
        """
        Return a boolean value depending if the state is a combination of states.

        Returns:
            bool: True if the state is a combination of states, False otherwise.
        """
        return len(self._states) > 0

    # alias
    has_states = is_combined_states

    def has_data(self):
        """Check if the state has data."""
        return self._data is not None
        # return len(self._states) == 0

    def has_space(self):
        """Check if the state has a space."""
        return self._space is not None

    def add(self, state):
        """
        Add a state or a list of states to the list of internal states. Useful when combining different states together.
        This shouldn't be called if this state has some data set to it.

        Args:
            state (State, list/tuple of State): state(s) to add to the internal list of states
        """
        if self.has_data():
            raise AttributeError("Undefined behavior: a state should be a combination of states or should contain "
                                 "some kind of data, but not both.")
        if isinstance(state, State):
            self._states.add(state)
        elif isinstance(state, collections.Iterable):
            for i, s in enumerate(state):
                if not isinstance(s, State):
                    raise TypeError("The item {} in the given list is not an instance of State".format(i))
                self._states.add(s)
        else:
            raise TypeError("The 'other' argument should be an instance of State, or an iterator over states.")

    # alias
    append = add
    extend = add

    def _read(self):
        """
        Read the state value. This has to be overwritten in the child class.
        """
        pass

    def read(self):
        """
        Read the state values from the simulator for each state, set it and return their values.
        """
        # if time to read
        if self._cnt % self.ticks == 0:

            # if multiple states, read each state
            if self.has_states():  # read each state
                for state in self.states:
                    state._read()
            else:  # else, read the current state
                self._read()

        # increment counter
        self._cnt += 1

        # return the data
        return self.data

    def _reset(self):
        """
        Reset the state. This has to be overwritten in the child class.
        """
        self._cnt = 0
        self._read()

    def reset(self):
        """
        Some states need to be reset. It returns the initial state.
        """
        self._cnt = 0

        # if multiple states, reset each state
        if self.has_states():
            for state in self.states:
                state._reset()
        else:  # else, reset this state
            self._reset()

        # return the first state data
        return self.data  # self.read()

    def max_dimension(self):
        """
        Return the maximum dimension.
        """
        return max(self.dimension)

    def total_size(self):
        """
        Return the total size of the combined state.
        """
        return sum(self.size)

    def has_discrete_values(self):
        """
        Does the state have discrete values?
        """
        if self._data is None:
            return [isinstance(state._space, gym.spaces.Discrete) for state in self._states]
        if isinstance(self._space, gym.spaces.Discrete):
            return [True]
        return [False]

    def is_discrete(self):
        """
        If all the states are discrete, then it is discrete.
        """
        return all(self.has_discrete_values())

    def has_continuous_values(self):
        """
        Does the state have continuous values?
        """
        if self._data is None:
            return [isinstance(state._space, gym.spaces.Box) for state in self._states]
        if isinstance(self._space, gym.spaces.Box):
            return [True]
        return [False]

    def is_continuous(self):
        """
        If one of the state is continuous, then the state is considered to be continuous.
        """
        return any(self.has_continuous_values())

    def bounds(self):
        """
        If the state is continuous, it returns the lower and higher bounds of the state.
        If the state is discrete, it returns the maximum number of discrete values that the state can take.

        Returns:
            list/tuple: list of bounds if multiple states, or bounds of this state
        """
        if self._data is None:
            return [state.bounds() for state in self._states]
        if isinstance(self._space, gym.spaces.Box):
            return self._space.low, self._space.high
        elif isinstance(self._space, gym.spaces.Discrete):
            return (self._space.n,)
        raise NotImplementedError

    def apply(self, fct):
        """
        Apply the given fct to the data of the state, and set it to the state.
        """
        self.data = fct(self.data)

    def contains(self, x):  # parameter dependent of the state
        """
        Check if the argument is within the range/bound of the state.
        """
        return self._space.contains(x)

    def sample(self, distribution=None):  # parameter dependent of the state (discrete and continuous distributions)
        """
        Sample some values from the state based on the given distribution.
        If no distribution is specified, it samples from a uniform distribution (default value).
        """
        if self.is_combined_states():
            return [state.sample() for state in self._states]
        if self._distribution is None:
            return
        else:
            pass
        raise NotImplementedError

    def add_noise(self, noise=None, replace=True):  # parameter dependent of the state
        """
        Add some noise to the state, and returns it.

        Args:
            noise (np.ndarray, fct): array to be added or function to be applied on the data
        """
        if self._data is None:
            # apply noise
            for state in self._states:
                state.add_noise(noise=noise)
        else:
            # add noise to the data
            noisy_data = self.data[0] + noise
            # clip such that the data is within the bounds
            self.data = noisy_data

    def normalize(self, normalizer=None, replace=True):  # parameter dependent of the state
        """
        Normalize using the state data using the provided normalizer.

        Args:
            normalizer (sklearn.preprocessing.Normalizer): the normalizer to apply to the data.
            replace (bool): if True, it will replace the `data` attribute by the normalized data.

        Returns:
            the normalized data
        """
        pass

    def fuse(self, other=None, axis=0):
        """
        Fuse the states that have the same shape together. The axis specified along which axis we concatenate the data.
        If multiple states with different shapes are present, the axis will be the one specified if possible, otherwise
        it will be min(dimension, axis).

        Examples:
            s0 = JntPositionState(robot)
            s1 = JntVelocityState(robot)
            s = s0 & s1
            print(s)
            print(s.shape)
            s = s0 + s1
            s.fuse()
            print(s)
            print(s.shape)
        """
        # check argument
        if not (other is None or isinstance(other, State)):
            raise TypeError("The 'other' argument should be None or another state.")

        # build list of all the states
        states = [self] if self.has_data() else self._states
        if other is not None:
            if other.has_data():
                states.append(other)
            else:
                states.extend(other._states)

        # check if only one state
        if len(states) < 2:
            return self  # do nothing

        # build the dictionary with key=dimension of shape, value=list of states
        dic = {}
        for state in states:
            dic.setdefault(len(state.data[0].shape), []).append(state)

        # traverse the dictionary and fuse corresponding shapes
        states = []
        for key, value in dic.items():
            if len(value) > 1:
                # fuse
                data = [state.data[0] for state in value]
                names = [state.name for state in value]
                s = State(data=np.concatenate(data, axis=min(axis, key)), name='+'.join(names))
                states.append(s)
            else:
                # only one state
                states.append(value[0])

        # return the fused state
        if len(states) == 1:
            return states[0]
        return State(states)

    def lookfor(self, class_type):
        """
        Look for the specified class type/name in the list of internal states, and returns it.

        Args:
            class_type (type, str): class type or name

        Returns:
            State: the corresponding instance of the State class
        """
        # if string, lowercase it
        if isinstance(class_type, str):
            class_type = class_type.lower()

        # if there is one state
        if self.has_data():
            if self.__class__ == class_type or self.__class__.__name__.lower() == class_type or self.name == class_type:
                return self

        # the state has multiple states, thus we go through each state
        for state in self.states:
            if state.__class__ == class_type or state.__class__.__name__.lower() == class_type or \
                    state.name == class_type:
                return state

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a representation string about the object."""
        if self._data is None:
            lst = [self.__class__.__name__ + '(']
            for state in self.states:
                lst.append('\t' + state.__str__() + ',')
            lst.append(')')
            return '\n'.join(lst)
        else:
            return '%s(%s)' % (self.name, self.data[0])

    # def __str__(self):
    #     """
    #     String to represent the state. Need to be provided by each child class.
    #     """
    #     if self._data is None:
    #         return [str(state) for state in self._states]
    #     return str(self)

    def __call__(self):
        """
        Compute/read the state and return it. It is an alias to the `self.read()` method.
        """
        return self.read()

    def __len__(self):
        """
        Return the total number of states contained in this class.

        Example::

            s1 = JntPositionState(robot)
            s2 = s1 + JntVelocityState(robot)
            print(len(s1)) # returns 1
            print(len(s2)) # returns 2
        """
        if self._data is None:
            return len(self._states)
        return 1

    def __iter__(self):
        """
        Iterator over the states.
        """
        if self.is_combined_states():
            for state in self._states:
                yield state
        else:
            yield self

    def __contains__(self, item):
        """
        Check if the state item(s) is(are) in the combined state. If the item is the data associated with the state,
        it checks that it is within the bounds.

        Args:
            item (State, list/tuple of state, type): check if given state(s) is(are) in the combined state

        Example:
            s1 = JointPositionState(robot)
            s2 = JointVelocityState(robot)
            s = s1 + s2
            print(s1 in s) # output True
            print(s2 in s1) # output False
            print((s1, s2) in s) # output True
        """
        # check type of item
        if not isinstance(item, (State, np.ndarray, type)):
            raise TypeError("Expecting a State, np.array, or a class type, instead got: {}".format(type(item)))

        # if class type
        if isinstance(item, type):
            # if there is one state
            if self.has_data():
                return self.__class__ == item
            # the state has multiple states, thus we go through each state
            for state in self.states:
                if state.__class__ == item:
                    return True
            return False

        # check if state item is in the combined state
        if self._data is None and isinstance(item, State):
            return item in self._states

        # check if state/data is within the bounds
        if isinstance(item, State):
            item = item.data

        # check if continuous
        # if self.is_continuous():
        #     low, high = self.bounds()
        #     return np.all(low <= item) and np.all(item <= high)
        # else: # discrete case
        #     num = self.bounds()[0]
        #     # check the size of data
        #     if item.size > 1: # array
        #         return (item.size < num)
        #     else: # one number
        #         return (item[0] < num)

        return self.contains(item)

    def __getitem__(self, key):
        """
        Get the corresponding item from the state(s)
        """
        # if one state, slice the corresponding state data
        if len(self._states) == 0:
            return self.data[0][key]
        # if multiple states
        if isinstance(key, int):
            # get one state
            return self._states[key]
        elif isinstance(key, slice):
            # get multiple states
            return State(self._states[key])
        else:
            raise TypeError("Expecting an int or slice for the key, but got instead {}".format(type(key)))

    def __setitem__(self, key, value):
        """
        Set the corresponding item/value to the corresponding key.

        Args:
            key (int, slice): index of the internal state, or index/indices for the state data
            value (State, int/float, array): value to be set
        """
        if self.is_combined_states():
            # set/move the state to the specified key
            if isinstance(value, State) and isinstance(key, int):
                self._states[key] = value
            else:
                raise TypeError("Expecting key to be an int, and value to be a state.")
        else:
            # set the value on the data directly
            self.data[0][key] = value

    def __add__(self, other):
        """
        Combine two different states together. In this special case, the operation is not commutable.
        This is the same as taking the union of the states.

        Args:
            other (State): another state

        Returns:
            State: the combined state

        Examples:
            s1 = JntPositionState(robot)
            s2 = JntVelocityState(robot)
            s = s1 + s2     # = State([JntPositionState(robot), JntVelocityState(robot)])

            s1 = State([JntPositionState(robot), JntVelocityState(robot)])
            s2 = State([JntPositionState(robot), LinkPositionState(robot)])
            s = s1 + s2     # = State([JntPositionState(robot), JntVelocityState(robot), LinkPositionState(robot)])
        """
        if not isinstance(other, State):
            raise TypeError("Expecting another state, instead got {}".format(type(other)))
        s1 = self._states if self._data is None else OrderedSet([self])
        s2 = other._states if other._data is None else OrderedSet([other])
        s = s1 + s2
        return State(s)

    def __iadd__(self, other):
        """
        Add a state to the current one.

        Args:
            other (State, list/tuple of State): other state

        Examples:
            s = State()
            s += JntPositionState(robot)
            s += JntVelocityState(robot)
        """
        if self._data is not None:
            raise AttributeError("The current class already has some data attached to it. This operation can not be "
                                 "applied in this case.")
        self.append(other)

    def __sub__(self, other):
        """
        Remove the other state(s) from the current state.

        Args:
            other (State): state to be removed.
        """
        if not isinstance(other, State):
            raise TypeError("Expecting another state, instead got {}".format(type(other)))
        s1 = self._states if self._data is None else OrderedSet([self])
        s2 = other._states if other._data is None else OrderedSet([other])
        s = s1 - s2
        if len(s) == 1:  # just one element
            return s[0]
        return State(s)

    def __isub__(self, other):
        """
        Remove one or several states from the combined state.

        Args:
            other (State): state to be removed.
        """
        if not isinstance(other, State):
            raise TypeError("Expecting another state, instead got {}".format(type(other)))
        if self._data is not None:
            raise RuntimeError("This operation is only available for a combined state")
        s = other._states if other._data is None else OrderedSet([other])
        self._states -= s

    def __and__(self, other):
        """
        Fuse two states together; only one data for the two states, instead of a data for each state as done
        when combining the states. All the states must have the same dimensions, and it fuses the data along
        the axis=0.

        Args:
            other: the other (combined) state

        Returns:
            State: the intersection of states

        Examples:
            s0 = JntPositionState(robot)
            s1 = JntVelocityState(robot)
            print(s0.shape)
            print(s1.shape)
            s = s0 + s1
            print(s.shape)  # prints [s0.shape, s1.shape]
            s = s0 & s1
            print(s.shape)  # prints np.concatenate((s0,s1)).shape
        """
        return self.fuse(other, axis=0)

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(states=self.states, data=self._data, space=self._space, name=self.name,
                              window_size=self.window_size, axis=self.axis, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        states = [copy.deepcopy(state, memo) for state in self.states]
        data = copy.deepcopy(self.window[0]) if self.has_data() else None
        space = copy.deepcopy(self._space)
        state = self.__class__(states=states, data=data, space=space, name=self.name, window_size=self.window_size,
                               axis=self.axis, ticks=self.ticks)

        memo[self] = state
        return state

    # def __invert__(self):
    #     """
    #     Return the
    #     :return:
    #     """
    #     pass
