#!/usr/bin/env python
"""Define most common rewards used in reinforcement learning and optimization.

A reward is defined as an objective that compliments/rewards a certain behavior.
The `Reward` class inherits from the `Objective` class.

To see the documentation of a certain reward in the python interpreter, just type:
```python
from reward import <Reward>
print(<Reward>.__doc__)
```

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
"""

import operator
import copy

# from pyrobolearn.rewards.objective import Objective
from pyrobolearn.states import *
from pyrobolearn.actions import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# class Reward(Objective):
class Reward(object):
    r"""Abstract `Reward` class which inherits from the `Objective` class, and is set to be maximized.
    Every classes that defines a reward inherits from this one. A reward is defined as an objective
    that compliments/rewards a certain behavior.

    A reward is defined as [1]:
    - r(s): given the state s, it returns the reward.
    - r(s,a): given the state s and action a, it returns the reward.
    - r(s,a,s'): given the state s, action a, and next state s', it returns the reward.

    Note:
        - In order to enable binary operators such as `add`, `multiply`, and so on, we can accomplish it using two
          different approaches; the functional [2] and `eval` approach.
            - Functional approach: we define for each operator, what functions to call and set them to the new created
                                   reward.
            - Eval approach: we build the string that represents the operations to carry out, and evaluate it when we
                             compute the reward.

    Examples:
        # create simulator
        sim = Simulator()

        # create world and load robot
        world = World(sim)
        robot = world.load(Robot())

        # create state, action, and policy
        state = JntPositionState(robot) + JntVelocityState(robot)
        action = JntPositionAction(robot)
        policy = Policy(state, action)

        # create reward
        reward = 2 * ForwardProgressReward(CoMState(robot)) + 4 * NotFallingReward(robot) + exp(FixedReward(-1))

        # create env
        env = Env(world, state, reward)

        # create and run task with the current policy
        task = Task(env, policy)
        task.run(num_steps=1000) # run loop between 'env' and 'policy' for num_steps

        # create algo to train the policy, and run it
        algo = Algo(task)
        algo.train()

        # run task with the trained policy
        task.run(num_steps=1000)

    References:
        [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998
        [2] https://turion.wordpress.com/2012/01/05/add-and-multiply-python-functions-operable-functions/
    """

    def __init__(self, state=None, action=None, rewards=None, range=(-np.infty, np.infty)):
        """
        Initialize the reward function.

        Args:
            state (None, State): state on which depends the reward function.
            action (None, Action): action on which depends the reward function.
            rewards (None, list of Reward): list of intern rewards.
            range (tuple of float, np.array of 2 float): A tuple corresponding to the min and max possible rewards.
                By default, it is [-infinity, infinity]. The computed reward is automatically clipped if it goes
                outside the range.
        """
        # super(Reward, self).__init__(maximize=True)

        self.state = state
        self.action = action
        self.rewards = rewards
        self.range = range

        # # create automatically binary operator methods
        # op_names = ['__add__', '__div__', '__floordiv__', '__iadd__', '__idiv__', '__ifloordiv__', '__imod__',
        #             '__imul__', '__ipow__', '__isub__', '__itruediv__',  '__mod__', '__mul__', '__pow__', '__radd__',
        #             '__rdiv__', '__rfloordiv__', '__rmod__', '__rmul__', '__rpow__', '__rsub__', '__rtruediv__',
        #             '__sub__', '__truediv__']
        # for name in op_names:
        #     # define binary operator method
        #     def wrapper(op):
        #         def binary_operator(self, other):
        #             # built the internal list of rewards
        #             rewards = self._rewards if self.hasRewards() else [self]
        #             if isinstance(other, Reward):
        #                 if other.hasRewards():
        #                     rewards.extend(other._rewards)
        #                 else:
        #                     rewards.append(other)
        #
        #             # create reward to return
        #             reward = Reward(rewards=rewards)
        #
        #             # replace the `reward._compute` by the corresponding function
        #             if isinstance(other, Reward): # callable
        #                 def compute():
        #                     return op(self(), other())
        #             else:
        #                 def compute():
        #                     return op(self(), other)
        #             reward._compute = compute
        #             return reward
        #         return binary_operator
        #
        #     op = getattr(operator, name) if not name.startswith('__r') else getattr(operator, name.replace('r','',1))
        #     setattr(self.__class__, name, wrapper(op))

        # # create automatically binary comparison operator methods
        # op_names = ['__eq__', '__ge__', '__gt__', '__le__', '__lt__', '__ne__']
        # for name in op_names:
        #     # define binary operator method
        #     def wrapper(op):
        #         def binary_operator(self, other):
        #             if isinstance(other, Reward): # callable
        #                 def compute():
        #                     return op(self(), other())
        #             else:
        #                 def compute():
        #                     return op(self(), other)
        #             return compute()
        #         return binary_operator
        #
        #     op = getattr(operator, name)
        #     setattr(self.__class__, name, wrapper(op))

        # # create automatically unary operator methods
        # op_names = ['__abs__', '__neg__', '__pos__']
        # for name in op_names:
        #     # define unary method
        #     def wrapper(op):
        #         def unary_operator(self):
        #             reward = copy.copy(self) # shallow copy
        #             def compute():
        #                 return op(self())
        #             reward._compute = compute
        #             return reward
        #         return unary_operator
        #
        #     op = getattr(operator, name)
        #     setattr(self.__class__, name, wrapper(op))

    ##############
    # Properties #
    ##############

    @property
    def state(self):
        """Return the state instance."""
        return self._state

    @state.setter
    def state(self, state):
        """Set the state."""
        if state is not None and not isinstance(state, State):
            raise TypeError("Expecting state to be None or an instance of State.")
        self._state = state

    @property
    def action(self):
        """Return the action instance."""
        return self._action

    @action.setter
    def action(self, action):
        """Set the action."""
        if action is not None and not isinstance(action, Action):
            raise TypeError("Expecting action to be None or an instance of Action.")
        self._action = action

    @property
    def rewards(self):
        """Return the inner rewards."""
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        """Set the inner rewards."""
        if rewards is None:
            rewards = []
        elif isinstance(rewards, collections.Iterable):
            for reward in rewards:
                if not isinstance(reward, Reward):
                    raise TypeError("Expecting a Reward instance for each item in the iterator.")
        else:
            if not isinstance(rewards, Reward):
                raise TypeError("Expecting rewards to be an instance of Reward.")
            rewards = [rewards]
        self._rewards = rewards

    @property
    def range(self):
        """Return the range of the reward function; a tuple corresponding to the min and max possible rewards."""
        return self._range

    @range.setter
    def range(self, range):
        """Set the range of the reward function; a tuple corresponding to the min and max possible rewards."""
        if range is None:
            range = (-np.infty, np.infty)
        if not isinstance(range, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given 'range' to be a tuple/list/np.array of 2 float, instead got: "
                            "{}".format(type(range)))
        if len(range) != 2:
            raise ValueError("Expecting the given 'range' to be a tuple/list/np.array of len(2), instead got a length "
                             "of: {}".format(len(range)))
        if range[0] > range[1]:
            raise ValueError("Expecting range[0] <= range[1], instead got the opposite: {}".format(range))
        self._range = np.array(range)

    ###########
    # Methods #
    ###########

    def has_rewards(self):
        """Check if there are inner rewards."""
        return len(self._rewards) > 0

    @staticmethod
    def is_maximized():
        """Check if it is maximized."""
        return True

    def reset(self):
        """Reset the rewards."""
        for reward in self.rewards:
            reward.reset()

    def _compute(self):  # **kwargs):
        """Compute the reward and return the scalar value. This has to be implemented in the child classes.

        Warnings: by default, *args and **kwargs are disabled as it could lead to several problems:
        1. As more and more rewards will become available, there might share the same argument name if the programmer
        is not careful, which could lead to bugs that are difficult to detect.
        2. It is better to provide the arguments during the initialization of the reward class. If the user has
        a variable that might change, create a class for that variable and in the corresponding reward's `compute`
        method, check what is its value.
        """
        raise NotImplementedError

    def compute(self):
        return np.clip(self._compute(), self.range[0], self.range[1])

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a representation string about the reward function."""
        if not self.rewards or self.rewards is None:
            return self.__class__.__name__
        else:
            lst = [reward.__str__() for reward in self.rewards]
            return ' + '.join(lst)

    def __call__(self):  # **kwargs):
        """Compute the reward function."""
        return self.compute()  # **kwargs)

    def __copy__(self):
        """Return a shallow copy of the reward function. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action, rewards=self.rewards, range=self.range)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the reward function. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        state = copy.deepcopy(self.state, memo)
        action = copy.deepcopy(self.action, memo)
        rewards = [copy.deepcopy(reward, memo) for reward in self.rewards]
        range = copy.deepcopy(self.range)
        reward = self.__class__(state=state, action=action, rewards=rewards, range=range)

        memo[self] = reward
        return reward

    # for unary and binary operators, see `__init__()` method.

    def __build_reward(self, other):
        """Build the list of inner rewards."""
        # built the internal list of rewards
        rewards = self._rewards if self.has_rewards() else [self]
        if isinstance(other, Reward):
            if other.has_rewards():
                rewards.extend(other._rewards)
            else:
                rewards.append(other)
        return Reward(rewards=rewards)

    @staticmethod
    def __get_operation(a, b, op):
        """Return the compute function that we called when calling the reward function.

        Args:
            a (float, int, Reward): first term in the operation: op(a, b)
            b (float, int, Reward): second term in the operation: op(a, b)
            op (types.BuiltinFunctionType): operator to be applied on the two given terms: op(a, b)
        """
        if isinstance(a, Reward):  # callable
            if isinstance(b, Reward):
                def compute():
                    return op(a(), b())
            else:
                def compute():
                    return op(a(), b)
        else:
            if isinstance(b, Reward):
                def compute():
                    return op(a, b())
            else:  # this condition should normally never be satisfied
                def compute():
                    return op(a, b)
        return compute

    @staticmethod
    def __build_range(a, b, op):
        """Build the range of the reward function.

        Args:
            a (float, int, Reward): first term in the operation: op(a, b)
            b (float, int, Reward): second term in the operation: op(a, b)
            op (types.BuiltinFunctionType): operator to be applied on the two given terms: op(a, b)
        """
        a = a.range if isinstance(a, Reward) else (a, a)
        b = b.range if isinstance(b, Reward) else (b, b)

        # check that you do not have a possible division or modulo by zero
        if op in {operator.__div__, operator.__floordiv__, operator.__truediv__, operator.__mod__}:
            if b[0] <= 0 <= b[1]:
                raise ValueError("Zero is between the lower and upper bound of the range of `other`. This is not "
                                 "accepted as it can lead to a division or modulo by zero.")

        # perform all possible combinations (Be careful with division or modulo by 0)
        operations = [op(a[0], b[0]), op(a[0], b[1]), op(a[1], b[0]), op(a[1], b[1])]

        # compute the lower and upper bounds
        low = np.min(operations)
        high = np.max(operations)

        # return the lower bound
        return low, high

    def __add__(self, other):
        """Return a reward that will add `self` and `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__add__)
        reward.range = self.__build_range(self, other, operator.__add__)
        return reward

    def __div__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__div__)
        reward.range = self.__build_range(self, other, operator.__div__)
        return reward

    def __floordiv__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__floordiv__)
        reward.range = self.__build_range(self, other, operator.__floordiv__)
        return reward

    def __iadd__(self, other):
        """Return a reward that will add `self` and `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__iadd__)
        reward.range = self.__build_range(self, other, operator.__add__)
        return reward

    def __idiv__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__idiv__)
        reward.range = self.__build_range(self, other, operator.__div__)
        return reward

    def __ifloordiv__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__ifloordiv__)
        reward.range = self.__build_range(self, other, operator.__floordiv__)
        return reward

    def __imod__(self, other):
        """Return a reward that will compute `self` modulo `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__imod__)
        reward.range = self.__build_range(self, other, operator.__mod__)
        return reward

    def __imul__(self, other):
        """Return a reward that will multiply `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__imul__)
        reward.range = self.__build_range(self, other, operator.__mul__)
        return reward

    def __ipow__(self, other):
        """Return a reward that will raise `self` to the power `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__ipow__)
        reward.range = self.__build_range(self, other, operator.__pow__)
        return reward

    def __isub__(self, other):
        """Return a reward that will subtract `other` from `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__isub__)
        reward.range = self.__build_range(self, other, operator.__sub__)
        return reward

    def __itruediv__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__itruediv__)
        reward.range = self.__build_range(self, other, operator.__truediv__)
        return reward

    def __mod__(self, other):
        """Return a reward that will compute `self` modulo `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__mod__)
        reward.range = self.__build_range(self, other, operator.__mod__)
        return reward

    def __mul__(self, other):
        """Return a reward that will multiply `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__mul__)
        reward.range = self.__build_range(self, other, operator.__mul__)
        return reward

    def __pow__(self, other):
        """Return a reward that will raise `self` to the power `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__pow__)
        reward.range = self.__build_range(self, other, operator.__pow__)
        return reward

    def __radd__(self, other):
        """Return a reward that will add `self` and `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__add__)
        reward.range = self.__build_range(self, other, operator.__add__)
        return reward

    def __rdiv__(self, other):
        """Return a reward that will divide `other` by `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__div__)
        reward.range = self.__build_range(other, self, operator.__div__)
        return reward

    def __rfloordiv__(self, other):
        """Return a reward that will divide `other` by `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__floordiv__)
        reward.range = self.__build_range(other, self, operator.__floordiv__)
        return reward

    def __rmod__(self, other):
        """Return a reward that will compute `other` modulo `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__mod__)
        reward.range = self.__build_range(other, self, operator.__mod__)
        return reward

    def __rmul__(self, other):
        """Return a reward that will multiply `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__mul__)
        reward.range = self.__build_range(self, other, operator.__mul__)
        return reward

    def __rpow__(self, other):
        """Return a reward that will raise `other` to the power `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__pow__)
        reward.range = self.__build_range(other, self, operator.__pow__)
        return reward

    def __rsub__(self, other):
        """Return a reward that will subtract `self` from `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__sub__)
        reward.range = self.__build_range(other, self, operator.__sub__)
        return reward

    def __rtruediv__(self, other):
        """Return a reward that will divide `other` by `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(other, self, operator.__truediv__)
        reward.range = self.__build_range(other, self, operator.__truediv__)
        return reward

    def __sub__(self, other):
        """Return a reward that will subtract `other` from `self`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__sub__)
        reward.range = self.__build_range(self, other, operator.__sub__)
        return reward

    def __truediv__(self, other):
        """Return a reward that will divide `self` by `other`."""
        reward = self.__build_reward(other)
        reward._compute = self.__get_operation(self, other, operator.__truediv__)
        reward.range = self.__build_range(self, other, operator.__truediv__)
        return reward

    # binary comparison operators
    def __eq__(self, other):
        compute = self.__get_operation(self, other, operator.__eq__)
        return compute()

    def __ge__(self, other):
        compute = self.__get_operation(self, other, operator.__ge__)
        return compute()

    def __gt__(self, other):
        compute = self.__get_operation(self, other, operator.__gt__)
        return compute()

    def __le__(self, other):
        compute = self.__get_operation(self, other, operator.__le__)
        return compute()

    def __lt__(self, other):
        compute = self.__get_operation(self, other, operator.__lt__)
        return compute()

    def __ne__(self, other):
        compute = self.__get_operation(self, other, operator.__ne__)
        return compute()

    # unary operators
    def __abs__(self):
        """Return the absolute value of the reward."""
        reward = copy.copy(self)  # shallow copy
        reward._compute = lambda: operator.__abs__(self())
        if self.range[0] <= 0 <= self.range[1]:  # the lower bound is negative while the upper bound is positive
            low, high = 0, self.range[1]
        elif self.range[0] < 0 and self.range[1] < 0:  # the lower and upper bounds are negative
            if self.range[0] < self.range[1]:
                low, high = -self.range[1], -self.range[0]
            else:
                low, high = -self.range[0], -self.range[1]
        else:  # the lower and upper bounds are positive
            low, high = self.range[0], self.range[1]
        reward.range = (low, high)
        return reward

    def __neg__(self):
        """Return the resulting reward by applying the unary negative operation on it."""
        reward = copy.copy(self)  # shallow copy
        reward._compute = lambda: operator.__neg__(self())
        reward.range = (-self.range[1], -self.range[0])
        return reward

    def __pos__(self):
        """Return the resulting reward by applying the unary positive operation on it."""
        reward = copy.copy(self)  # shallow copy
        reward._compute = lambda: operator.__pos__(self())
        reward.range = self.range
        return reward


######################################
# mathematical operations on rewards #
######################################
# import math
# import numpy as np
# names = []
# name_lst = [name for name in dir(math) if '__' not in name and name in dir(np)]
# name_lst.remove('e')
# for name in name_lst:
#     op = getattr(np, name)
#     if callable(op):
#         try:
#             op(1)
#         except (ValueError, TypeError) as e:
#             pass
#         else:
#             names.append(name)

# # define the following mathematical operations automatically
# names = ['ceil', 'cos', 'cosh', 'degrees', 'exp', 'expm1', 'fabs', 'floor', 'frexp', 'isinf', 'isnan', 'log', 'log10',
#          'log1p', 'modf', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc']
# for name in names:
#     def wrapper(op):
#         def fct(x):
#             if callable(x):
#                 y = copy.copy(x)  # shallow copy
#                 def f():
#                     return op(x())
#                 y._compute = f
#                 return y
#             else:
#                 return op(x)
#         return fct
#
#     globals()[name] = wrapper(getattr(np, name))

def ceil(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.ceil(x())
        y.range = (np.ceil(x.range[0]), np.ceil(x.range[1]))
        return y
    else:
        return np.ceil(x)


def cos(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.cos(x())
        y.range = (np.cos(x.range[0]), np.cos(x.range[1]))
        return y
    else:
        return np.cos(x)


def cosh(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.cosh(x())
        y.range = (np.cosh(x.range[0]), np.cosh(x.range[1]))
        return y
    else:
        return np.cosh(x)


def degrees(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.degrees(x())
        y.range = (np.degrees(x.range[0]), np.degrees(x.range[1]))
        return y
    else:
        return np.degrees(x)


def exp(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.exp(x())
        y.range = (np.exp(x.range[0]), np.exp(x.range[1]))
        return y
    else:
        return np.exp(x)


def expm1(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.expm1(x())
        y.range = (np.expm1(x.range[0]), np.expm1(x.range[1]))
        return y
    else:
        return np.expm1(x)


def floor(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.floor(x())
        y.range = (np.floor(x.range[0]), np.floor(x.range[1]))
        return y
    else:
        return np.floor(x)


def frexp(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.frexp(x())
        y.range = (np.frexp(x.range[0]), np.frexp(x.range[1]))
        return y
    else:
        return np.frexp(x)


def isinf(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.isinf(x())
        y.range = x.range
        return y
    else:
        return np.isinf(x)


def isnan(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.isnan(x())
        y.range = x.range
        return y
    else:
        return np.isnan(x)


def log(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.log(x())
        y.range = (np.log(x.range[0]), np.log(x.range[1]))
        return y
    else:
        return np.log(x)


def log10(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.log10(x())
        y.range = (np.log10(x.range[0]), np.log10(x.range[1]))
        return y
    else:
        return np.log10(x)


def log1p(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.log1p(x())
        y.range = (np.log1p(x.range[0]), np.log1p(x.range[1]))
        return y
    else:
        return np.log1p(x)


# def modf(x):
#     if isinstance(x, Reward):
#         y = copy.copy(x)  # shallow copy
#         y._compute = lambda: np.modf(x())
#         y.range = x.range
#         return y
#     else:
#         return np.modf(x)


def radians(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.radians(x())
        y.range = (np.radians(x.range[0]), np.radians(x.range[1]))
        return y
    else:
        return np.radians(x)


def sin(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.sin(x())
        y.range = (np.sin(x.range[0]), np.sin(x.range[1]))
        return y
    else:
        return np.sin(x)


def sinh(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.sinh(x())
        y.range = (np.sinh(x.range[0]), np.sinh(x.range[1]))
        return y
    else:
        return np.sinh(x)


def sqrt(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.sqrt(x())
        y.range = (np.sqrt(x.range[0]), np.sqrt(x.range[1]))
        return y
    else:
        return np.sqrt(x)


def tan(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.tan(x())
        y.range = (np.tan(x.range[0]), np.tan(x.range[1]))
        return y
    else:
        return np.tan(x)


def tanh(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.tanh(x())
        y.range = (np.tanh(x.range[0]), np.tanh(x.range[1]))
        return y
    else:
        return np.tanh(x)


def trunc(x):
    if isinstance(x, Reward):
        y = copy.copy(x)  # shallow copy
        y._compute = lambda: np.trunc(x())
        y.range = (np.trunc(x.range[0]), np.trunc(x.range[1]))
        return y
    else:
        return np.trunc(x)
