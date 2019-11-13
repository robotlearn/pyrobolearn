#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define various initial state generators.

The initial state generator generates the initial state which is returned by the environment when calling
`env.reset()`. Note that the state can be generated in a deterministic manner or randomly based on a distribution.

Dependencies:
    - `pyrobolearn.states`

See Also:
    - `pyrobolearn.envs`
"""

import queue
import numpy as np
from abc import ABCMeta

from pyrobolearn.states import State

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class StateGenerator(object):
    r"""Initial State Generator

    Initialize the state which will be given as the first state by the environment when calling ``env.reset()``.
    If for instance, the state consists of joint positions and velocities, we can generate them from a distribution
    that is given or learned from data.

    The `state generator` is tightly coupled with a `state` object.

    Sometimes a mapping between different states is necessary. For instance, the state generator might generate
    human joint states that need first to be mapped to robot joint states in order to initialize the robot.
    In this example, a kinematic mapping which is modeled mathematically or learned need to be provided additionally.
    This is particularly significant as robot data are lacking, while human data is pretty abundant.
    The mapping function has to return a `state` object.
    """

    def __init__(self, state, fct=None):
        """Initialize the state generator.

        Args:
            state (State): state instance.
            fct (callable, None): callback function to be called after generating the data.
        """
        self.state = state
        self.fct = fct if callable(fct) else None

    @property
    def state(self):
        """Return the state instance."""
        return self._state

    @state.setter
    def state(self, state):
        """Set the state."""
        if not isinstance(state, State):
            raise TypeError("Expecting the given state to be an instance of `State`, instead got: "
                            "{}".format(type(state)))
        self._state = state

    def generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        # generate the data
        data = self._generate(set_data=set_data, reset_state=reset_state)

        # call the user function
        if self.fct is not None:
            self.fct()

        return data

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        raise NotImplementedError

    # def __repr__(self):
    #     return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self, set_data=True, reset_state=True):
        """Call the generator."""
        return self.generate(set_data=set_data, reset_state=reset_state)


class FixedStateGenerator(StateGenerator):
    r"""Fixed Initial State Generator

    This generator returns the same initial state each time it is called.
    """

    def __init__(self, state, data=None, fct=None):
        """Initialize the fixed state generator.

        Args:
            state (State): state instance.
            data (int, float, list[float/int] np.array[float/int], None): initial data. If None, it will get the data
              from the given state, and set this last one as the initial data.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(FixedStateGenerator, self).__init__(state, fct=fct)
        if data is None:
            data = self.state.data
        self.initial_data = data

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        if set_data:
            self.state.data = self.initial_data
        if reset_state:
            self.state.reset()
        return self.initial_data


class QueueStateGenerator(StateGenerator):
    r"""Abstract Queue state generator
    """

    def __init__(self, state, queue, fct=None):
        """
        Initialize the queue state generator.

        Args:
            state (State): initial state.
            queue (queue.Queue): queue instance which contains the data.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(QueueStateGenerator, self).__init__(state, fct=fct)
        self.queue = queue
        self.initial_data = self.state.data

    @property
    def queue(self):
        """Return the queue."""
        return self._queue

    @queue.setter
    def queue(self, q):
        """Set the queue instance."""
        if not isinstance(q, queue.Queue):
            raise TypeError("Expecting the given queue to be an instance of `queue.Queue`, instead got: "
                            "{}".format(type(q)))
        self._queue = q

    def put(self, item, block=False, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot
        is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else raise
        the Full exception ('timeout' is ignored in that case).
        """
        if not self.queue.full():
            self.queue.put(item, block=block, timeout=timeout)

    # alias
    add = put

    def get(self, block=False, timeout=None):
        """Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until an item is
        available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately available, else raise the Empty exception
        ('timeout' is ignored in that case).
        """
        if self.queue.empty():
            return self.initial_data
        return self.queue.get(block=block, timeout=timeout)

    # alias
    pop = get

    def empty(self):
        """Return True if the queue is empty, False otherwise (not reliable!)."""
        return self.queue.empty()

    def full(self):
        """Return True if the queue is full, False otherwise (not reliable!)."""
        return self.queue.full()

    def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        return self.queue.qsize()

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        data = self.get()
        if isinstance(data, State):
            data = data.data
        if set_data:
            self.state.data = data
        if reset_state:
            self.state.reset()
        return data

    def __len__(self):
        """Return the size of the queue."""
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


class FIFOQueueStateGenerator(QueueStateGenerator):
    r"""FIFO Queue Initial State Generator

    Generate the initial state from a FIFO queue. If the queue is empty returns the default initial state.
    The queue is filled by the user during training.
    """

    def __init__(self, state, maxsize=0, fct=None):
        """Initialize the FIFO queue state generator.

        Args:
            state (State): state instance.
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
            fct (callable, None): callback function to be called after generating the data.
        """
        q = queue.Queue(maxsize)
        super(FIFOQueueStateGenerator, self).__init__(state, queue=q, fct=fct)


class LIFOQueueStateGenerator(QueueStateGenerator):
    r"""LIFO Queue Initial State Generator

    Generate the initial state from a LIFO queue. If the queue is empty returns the default initial state.
    The queue is filled by the user during training.
    """

    def __init__(self, state, maxsize=0, fct=None):
        """Initialize the LIFO queue state generator.

        Args:
            state (State): state instance.
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
            fct (callable, None): callback function to be called after generating the data.
        """
        q = queue.LifoQueue(maxsize)
        super(LIFOQueueStateGenerator, self).__init__(state, queue=q, fct=fct)


class PriorityQueueStateGenerator(QueueStateGenerator):
    r"""Priority Queue Initial State Generator

    Generate the initial state from a priority queue filled by the user. If empty, it returns the default initial
    state. The queue can be filled for instance with states that have high/low uncertainty, or high/low rewards.

    The queue has a limited capacity, and can be used to include states from which the agent/policy performed
    poorly during the training.
    """

    def __init__(self, state, maxsize=0, ascending=True, fct=None):
        """Initialize the priority queue state generator.

        Args:
            state (State): state instance.
            maxsize (int): maximum size of the queue. If :attr:`maxsize` is <= 0, the queue size is infinite.
            ascending (bool): if True, the item with the lowest priority will be the first one to be retrieved.
            fct (callable, None): callback function to be called after generating the data.
        """
        q = queue.PriorityQueue(maxsize)
        super(PriorityQueueStateGenerator, self).__init__(state, queue=q, fct=fct)
        self.ascending = ascending

    def get(self, block=False, timeout=None):
        """Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until an item is
        available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately available, else raise the Empty exception
        ('timeout' is ignored in that case).
        """
        if self.queue.empty():
           return self.initial_data
        item = self.queue.get(block=block, timeout=timeout)
        return item[1]

    def put(self, item, block=False, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default), block if necessary until a free slot
        is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot is immediately available, else raise
        the Full exception ('timeout' is ignored in that case).
        """
        if not self.queue.full():
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Expecting the item to be a tuple of length 2 with (priority number, data), instead "
                                "got: {}".format(item))
            if not self.ascending:
                item = (-item[0], item[1])
            self.queue.put(item, block=block, timeout=timeout)

    # aliases
    add = put
    pop = get


class StateDistributionGenerator(StateGenerator):
    r"""Initial State Distribution Generator

    The initial states :math:`s` are generated by a probability distribution :math:`p(s)`, that is :math:`s \sim p(s)`.
    The probability distribution can be learned using generative models.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, seed=None, fct=None):
        """Initialize the state distribution generator.

        Args:
            state (State): state instance.
            seed (None, int): random seed.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(StateDistributionGenerator, self).__init__(state, fct=fct)
        self.seed = seed

    @property
    def seed(self):
        """Return the random seed."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the random seed

        Args:
            seed (int): random seed
        """
        if seed is not None and not isinstance(seed, int):
            raise TypeError("Expecting the given 'seed' to be an integer, instead got: {}".format(type(seed)))
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed


class UniformStateGenerator(StateDistributionGenerator):
    r"""Uniform Initial State Generator

    The initial states are generated by a uniform distribution. If no upper/lower limits are specified, the limits
    will be set to be the range of the states.
    """
    
    def __init__(self, state, low=None, high=None, seed=None, fct=None):
        """Initialize the state distribution generator.

        Args:
            state (State): state instance.
            low (None, float, np.array, list of np.array): lower bound
            high (None, float, np.array, list of np.array): upper bound
            seed (None, int): random seed.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(UniformStateGenerator, self).__init__(state, seed=seed, fct=fct)
        self.low = low
        self.high = high

    @property
    def low(self):
        """Return the lower bound."""
        return self._low

    @low.setter
    def low(self, low):
        """Set the lower bound."""
        if low is None:
            low = [-np.infty] * len(self.state)
        elif isinstance(low, (int, float)):
            low = [low] * len(self.state)
        elif isinstance(low, (list, tuple, np.ndarray)):
            if len(low) != len(self.state):
                raise ValueError("The lower bound doesn't have the same size as the number of states; len(low) = {} "
                                 "and len(state) = {}".format(len(low), len(self.state)))
        else:
            raise TypeError("Expecting the 'low' bound to be an int, float, or list/tuple/np.array of int/float, "
                            "instead got: {}".format(type(low)))
        self._low = low

    @property
    def high(self):
        """Return the upper bound."""
        return self._high

    @high.setter
    def high(self, high):
        """Set the higher bound."""
        if high is None:
            high = [-np.infty] * len(self.state)
        elif isinstance(high, (int, float)):
            high = [high] * len(self.state)
        elif isinstance(high, (list, tuple, np.ndarray)):
            if len(high) != len(self.state):
                raise ValueError("The higher bound doesn't have the same size as the number of states; len(high) = {} "
                                 "and len(state) = {}".format(len(high), len(self.state)))
        else:
            raise TypeError("Expecting the 'high' bound to be an int, float, or list/tuple/np.array of int/float, "
                            "instead got: {}".format(type(high)))
        self._high = high

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        # spaces = self.state.space
        # data = [space.sample() for space in spaces]
        # for idx, datum, low, high in enumerate(zip(data, self.low, self.high)):
        #     data[idx] = np.clip(datum, low, high)

        data = [np.random.uniform(low=low, high=high, size=state.total_size())
                for state, low, high in zip(self.state, self.low, self.high)]
        if set_data:
            self.state.data = data
            print("Generate: {}".format(self.state.data))
        if reset_state:
            self.state.reset()
        return data


class NormalStateGenerator(StateDistributionGenerator):
    r"""Normal Initial State Generator

    The initial states are generated by a normal distribution, where the mean and standard deviation are specified.
    The states are then truncated / clipped to be inside their corresponding range.
    """

    def __init__(self, state, mean=0, covariance=1., seed=None, fct=None):
        """
        Initialize the Normal state generator.

        Args:
            state (State): state instance.
            mean (int, float, np.array[float[N]]): mean.
            scale (int, float, np.array[float[N]]): covariance matrix or variance vector.
            seed (None, int): random seed.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(NormalStateGenerator, self).__init__(state, seed=seed, fct=fct)
        self.mean = mean
        self.covariance = covariance

    @property
    def mean(self):
        """Return the mean vector."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        """Set the mean vector."""
        if mean is None:
            mean = [0.] * len(self.state)
        elif isinstance(mean, (int, float)):
            mean = [mean] * len(self.state)
        elif isinstance(mean, (list, tuple, np.ndarray)):
            if len(mean) != len(self.state):
                raise ValueError("The mean vector doesn't have the same size as the number of states; len(mean) = {} "
                                 "and len(state) = {}".format(len(mean), len(self.state)))
        else:
            raise TypeError("Expecting the mean vector to be an int, float, or list/tuple/np.array of int/float, "
                            "instead got: {}".format(type(mean)))
        self._mean = np.asarray(mean)

    @property
    def covariance(self):
        """Return the variance vector and covariance matrix."""
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        """Set the variance vector or covariance matrix."""
        if covariance is None:
            covariance = [1.] * len(self.state)
        elif isinstance(covariance, (int, float)):
            covariance = [covariance] * len(self.state)
        elif isinstance(covariance, (list, tuple, np.ndarray)):
            if len(covariance) != len(self.state):
                raise ValueError("The variance vector or covariance matrix doesn't have the same size as the number of "
                                 "states; len(covariance) = {} and len(state) = {}".format(len(covariance),
                                                                                           len(self.state)))
        else:
            raise TypeError("Expecting the variance vector or covariance matrix to be an int, float, or list/tuple/"
                            "np.array of int/float, but instead got: {}".format(type(covariance)))
        self._covariance = np.asarray(covariance)

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """

        if self.covariance.ndim == 2:
            data = [np.random.multivariate_normal(self.mean, self.covariance)]
        else:
            data = [np.random.normal(self.mean, scale=np.sqrt(self.covariance))]

        if set_data:
            self.state.data = data
            # print("Generate: {}".format(self.state.data))
        if reset_state:
            self.state.reset()
        return data


class GenerativeStateGenerator(StateGenerator):
    r"""Generative Initial State Generator

    This uses a generative model that has been trained to learn a distribution to generate the initial states.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, model, fct=None):
        """
        Initialize the Generative initial state generator.

        Args:
            state (State): state instance.
            model (Model): generative model instance.
            fct (callable, None): callback function to be called after generating the data.
        """
        super(GenerativeStateGenerator, self).__init__(state, fct=fct)
        self.model = model


class VAEStateGenerator(GenerativeStateGenerator):
    r"""Variational Autoencoder (VAE) Initial State Generator

    This uses the decoder a pretrained VAE to generate initial states.
    """

    def __init__(self, state, model, fct=None):
        super(VAEStateGenerator, self).__init__(state, model, fct=fct)

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        pass


class GANStateGenerator(GenerativeStateGenerator):
    r"""Generative Adversarial Network (GAN) Initial State Generator

    This uses the generator of a trained GAN model to generate similar states.
    """

    def __init__(self, state, model, distribution=None, mapping=None, fct=None):
        """
        Initialize the GAN initial state generator.

        Args:
            state: states that need to be generated
            model: GAN or generator of GAN
            distribution: distribution over the noise vector
            mapping:
        """
        # checking and setting the model
        if isinstance(model, GAN):
            self.generator = model.get_generator()
        elif isinstance(model, Generator):
            self.generator = model
        else:
            raise TypeError("The `model` parameter should be an instance of GAN or Generator.")

        # checking and setting the distribution
        if distribution is None:
            # create normal distribution with dimension of the generator input
            pass
        else:
            if not isinstance(distribution, Distribution):
                raise TypeError("The given `distribution` is not an instance of Distribution.")

        self.distribution = distribution

        # setting mapping
        self.mapping = mapping

        super(GANStateGenerator, self).__init__(state, model, fct=fct)

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        noise_vector = self.distribution.sample()
        states = self.generator(noise_vector)
        if self.mapping is not None:
            return self.mapping(states)
        return states


class GMMStateGenerator(GenerativeStateGenerator):
    r"""Gaussian Mixture Model Initial State Generator

    This uses a pretrained GMM to generate the states.
    """

    def __init__(self, state, model, fct=None):
        super(GMMStateGenerator, self).__init__(state, model, fct=fct)

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        pass


class UncertaintyStateGenerator(StateGenerator):
    r"""State generator that exploits the uncertainty of initial states.
    """
    __metaclass__ = ABCMeta
    pass


class BOStateGenerator(UncertaintyStateGenerator):
    r"""State generator based on Bayesian Optimization.

    We use Bayesian Optimization to generate the initial states.
    """
    __metaclass__ = ABCMeta
    pass


class AEBOStateGenerator(GenerativeStateGenerator):
    r"""AutoEncoder (AE) - Bayesian Optimization (BO) Initial State Generator

    Using a pretrained AE on plausible states, and keeping the decoder allows us to explore in the lower dimensional
    state space using BO (GP). The BO will provide the reduced state vector based on the uncertainty / objective
    fct value. Then, the outputted vector can be fed to the decoder which will return the corresponding high-
    dimensional state.

    In addition, we fix a certain capacity to the kernel matrix of the GP underlying the BO. If when inserting a
    new (low-dimensional) state, the capacity is exceeded, the oldest state is removed from the kernel to allow
    the incoming state.

    If the states have a certain range, we use the encoder part to get the corresponding low-dimensional state limits.
    The exploration will then be carried out in the hyperrectangle formed by these 2 reduced state vector limits.
    """

    def __init__(self, state, model, kernel_capacity=100, fct=None):
        """
        Initialize the autoencoder + bayesian optimization initial state generator.

        Args:
            state (State): state instance.
            model (Model): autoencoder model instance.
            kernel_capacity (int):
        """
        super(AEBOStateGenerator, self).__init__(state, model)

    def _generate(self, set_data=True, reset_state=True):
        """Generate the state.

        Args:
            set_data (bool): If True, it will set the generated data to the state.
            reset_state (bool): If True, it will reset the state with the generated data (if `set_data` has been set
                to True).

        Returns:
            (list of) np.array: state data
        """
        pass


# # Tests
# if __name__ == '__main__':
#     from pyrobolearn.states import AbsoluteTimeState, CumulativeTimeState
#
#     s = AbsoluteTimeState() + CumulativeTimeState()
#     s = CumulativeTimeState()
#     s.data = [2.]
#     data = s.data
#     print("Initial state: {}".format(data))
#
#     for _ in range(3):
#         s()
#         s.data = data
#         print(s.data)
#         print(data)
