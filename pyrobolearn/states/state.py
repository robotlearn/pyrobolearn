#!/usr/bin/env python
"""Define the State class.

This file defines the `State` class, which is returned by the environment, and given as an input to several
models such as policies/controllers, dynamic transition functions, value estimators, reward/cost function, and so on.
"""

import numpy as np
import collections
from abc import ABCMeta, abstractmethod
import gym

from pyrobolearn.utils.data_structures.orderedset import OrderedSet


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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

    def __init__(self, states=(), data=None, space=None, name=None):
        """
        Initialize the state. The state contains some kind of data, or is a state combined of other states.

        Args:
            states (list/tuple of State): list of states to be combined together (if given, we can not specified data)
            data (np.ndarray): data associated to this state
            space (gym.space): space associated with the given data

        Warning:
            Both arguments can not be provided to the state.
        """
        # Check arguments
        if states is None:
            states = tuple()

        if not isinstance(states, (list, tuple, set, OrderedSet)):
            # # TODO: should check that states is a list of state, however O(N)
            # if data is None: # this is in the case someone calls `State(data)`
            #     data = states
            # else:
            raise TypeError("Expecting a list, tuple, or (ordered) set of states.")
        if len(states) > 0 and data is not None:
            raise ValueError("Please specify only one of the argument `states` xor `data`, but not both.")

        # # Check if data is given
        # if data is not None:
        #     if not isinstance(data, np.ndarray):
        #         if isinstance(data, (list, tuple)):
        #             data = np.array(data)
        #         elif isinstance(data, (int, float)):
        #             data = np.array([data])
        #         else:
        #             raise TypeError("Expecting a numpy array, a list/tuple of int/float, or an int/float for 'data'")
        #
        # if isinstance(states, collections.Iterable):
        #     if len(states) > 0 and data is not None: # check if both the states and data are specified
        #         raise ValueError("Please specify only one of the argument `states` xor `data`, but not both.")
        #
        #     #
        #     for state in states:
        #         if not isinstance(state, State):
        #             if data is None: # in the case, someone calls `State(data)`
        #                 data = states
        #                 break
        #             else:
        #                 raise ValueError("Please specify only one of the argument `states` xor `data`, but not both.")

        # Check if data is given
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
        self._space = space
        self._distribution = None  # for sampling
        self._normalizer = None
        self._noiser = None  # for noise
        self._name = name

        # create ordered set which is useful if this state is a combination of multiple states
        self._states = OrderedSet()
        if self._data is None:
            self.add(states)

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
        if self.hasData():
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
        if self.hasData():
            return [self._data]
        return [state._data for state in self._states]

    @data.setter
    def data(self, data):
        """
        Set the data associated to this particular state, or the combined data associated to each state.
        Each data will be clipped if outside the range/bounds of the corresponding state.

        Args:
            data: the data to set
        """
        # one state: change the data
        if self.hasData():
            if not isinstance(data, np.ndarray):
                if isinstance(data, (list, tuple)):
                    data = np.array(data)
                elif isinstance(data, (int, float)):
                    data = data * np.ones(self._data.shape)
                else:
                    raise TypeError("Expecting a numpy array, a list/tuple of int/float, or an int/float for 'data'")
            if self._data.shape != data.shape:
                raise ValueError("The given data does not have the same shape as previously.")

            # clip the value using the space
            if self.hasSpace():
                if self.isContinuous():  # continuous case
                    low, high = self._space.low, self._space.high
                    data = np.clip(data, low, high)
                else:  # discrete case
                    n = self._space.n
                    if data.size == 1:
                        data = np.clip(data, 0, n)
            self._data = data

        else:  # combined state
            if not isinstance(data, collections.Iterable):
                raise TypeError("data is not an iterator")
            if len(self._states) != len(data):
                raise ValueError("The number of states is different from the number of data segments")
            for state, d in zip(self._states, data):
                state.data = d

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
    def space(self):
        """
        Get the corresponding space.
        """
        if self.hasSpace():
            return [self._space]
        return [state._space for state in self._states]

    @space.setter
    def space(self, space):
        """
        Set the corresponding space. This can only be used one time!
        """
        if self.hasData() and not self.hasSpace() and \
                isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):
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
        if not isinstance(name, str):
            raise TypeError("Expecting the name to be a string.")
        self._name = name

    @property
    def shape(self):
        """
        Return the shape of each state. Some states, such as camera states have more than 1 dimension.
        """
        return [d.shape for d in self.data]

    @property
    def size(self):
        """
        Return the size of each state.
        """
        return [d.size for d in self.data]

    @property
    def dimension(self):
        """
        Return the dimension (length of shape) of each state.
        """
        return [len(d.shape) for d in self.data]

    @property
    def distribution(self):
        """
        Get the current distribution used when sampling the state
        """
        pass

    @distribution.setter
    def distribution(self, distribution):
        """
        Set the distribution to the state.
        """
        # check if distribution is discrete/continuous
        pass

    ###########
    # Methods #
    ###########

    def isCombinedState(self):
        """
        Return a boolean value depending if the state is a combination of states.

        Returns:
            bool: True if the state is a combination of states, False otherwise.
        """
        return len(self._states) > 0

    # alias
    hasStates = isCombinedState

    def hasData(self):
        return self._data is not None

    def hasSpace(self):
        return self._space is not None

    def add(self, state):
        """
        Add a state or a list of states to the list of internal states. Useful when combining different states together.
        This shouldn't be called if this state has some data set to it.

        Args:
            state (State, list/tuple of State): state(s) to add to the internal list of states
        """
        if self.hasData():
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
        pass

    def read(self):
        """
        Read the state values from the simulator for each state, set it and return their values.
        This has to be overwritten by the child class.
        """
        if self.hasData():  # read the current state
            self._read()
        else:  # read each state
            for state in self.states:
                state._read()

        # return the data
        return self.data

    def _reset(self):
        pass

    def reset(self):
        """
        Some states need to be reset. It returns the initial state.
        This needs to be overwritten by the child class.

        Returns:
            initial state
        """
        if self.hasData():  # reset the current state
            self._reset()
        else:  # reset each state
            for state in self.states:
                state._reset()

        # return the first state data
        return self.read()

    def maxDimension(self):
        """
        Return the maximum dimension.
        """
        return max(self.dimension)

    def totalSize(self):
        """
        Return the total size of the combined state.
        """
        return sum(self.size)

    def hasDiscreteValues(self):
        """
        Does the state have discrete values?
        """
        if self._data is None:
            return [isinstance(state._space, gym.spaces.Discrete) for state in self._states]
        if isinstance(self._space, gym.spaces.Discrete):
            return [True]
        return [False]

    def isDiscrete(self):
        """
        If all the states are discrete, then it is discrete.
        """
        return all(self.hasDiscreteValues())

    def hasContinuousValues(self):
        """
        Does the state have continuous values?
        """
        if self._data is None:
            return [isinstance(state._space, gym.spaces.Box) for state in self._states]
        if isinstance(self._space, gym.spaces.Box):
            return [True]
        return [False]

    def isContinuous(self):
        """
        If one of the state is continuous, then the state is considered to be continuous.
        """
        return any(self.hasContinuousValues())

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
            return (self._space.low, self._space.high)
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
        if self.isCombinedState():
            return [state.sample() for state in self._states]
        if self._distribution is None:
            return
        else:
            pass
        raise NotImplementedError

    def addNoise(self, noise=None, replace=True):  # parameter dependent of the state
        """
        Add some noise to the state, and returns it.

        Args:
            noise (np.ndarray, fct): array to be added or function to be applied on the data
        """
        if self._data is None:
            # apply noise
            for state in self._states:
                state.addNoise(noise=noise)
        else:
            # add noise to the data
            noisy_data = self.data + noise
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
        states = [self] if self.hasData() else self._states
        if other is not None:
            if other.hasData():
                states.append(other)
            else:
                states.extend(other._states)

        # check if only one state
        if len(states) < 2:
            return self  # do nothing

        # build the dictionary with key=dimension of shape, value=state
        dic = {}
        for state in states:
            dic.setdefault(len(state._data.shape), []).append(state)

        # traverse the dictionary and fuse corresponding shapes
        states = []
        for key, value in dic.items():
            if len(value) > 1:
                # fuse
                data = [state._data for state in value]
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
        """
        if self.hasData():
            return None
        for state in self.states:
            if state.__class__ == class_type:
                return state

    ########################
    # Operator Overloading #
    ########################

    def __repr__(self):
        if self._data is None:
            lst = [self.__class__.__name__ + '(']
            for state in self.states:
                lst.append('\t' + state.__repr__() + ',')
            lst.append(')')
            return '\n'.join(lst)
        else:
            return '%s(%s)' % (self.name, self._data)

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
        if self.isCombinedState():
            for state in self._states:
                yield state
        else:
            yield self

    def __contains__(self, item):
        """
        Check if the state item(s) is(are) in the combined state. If the item is the data associated with the state,
        it checks that it is within the bounds.

        Args:
            item (State, list/tuple of state): check if given state(s) is(are) in the combined state

        Example:
            s1 = JntPositionState(robot)
            s2 = JntVelocityState(robot)
            s = s1 + s2
            print(s1 in s) # output True
            print(s2 in s1) # output False
            print((s1, s2) in s) # output True
        """
        # check type of item
        if not isinstance(item, (State, np.ndarray)):
            raise TypeError("Expecting a state or numpy array.")

        # check if state item is in the combined state
        if self._data is None and isinstance(item, State):
            return (item in self._states)

        # check if state/data is within the bounds
        if isinstance(item, State):
            item = item.data

        # check if continuous
        # if self.isContinuous():
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
            return self._data[key]
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
        if self.isCombinedState():
            # set/move the state to the specified key
            if isinstance(value, State) and isinstance(key, int):
                self._states[key] = value
            else:
                raise TypeError("Expecting key to be an int, and value to be a state.")
        else:
            # set the value on the data directly
            self._data[key] = value

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
        :param other:
        :return:
        """
        if not isinstance(other, State):
            raise TypeError("Expecting another state, instead got {}".format(type(other)))
        s1 = self._states if self._data is None else OrderedSet([self])
        s2 = other._states if other._data is None else OrderedSet([other])
        s = s1 - s2
        if len(s) == 1: # just one element
            return s[0]
        return State(s)

    def __isub__(self, other):
        """
        Remove one or several states from the combined state.

        Args:
            other:
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

    # def __invert__(self):
    #     """
    #     Return the
    #     :return:
    #     """
    #     pass
