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
import collections

# from objective import Objective
from pyrobolearn.states import *
from pyrobolearn.actions import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
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

    def __init__(self, state=None, action=None, rewards=None):
        # super(Reward, self).__init__(maximize=True)

        self.state = state
        self.action = action
        self.rewards = rewards

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
        #             # replace the `reward.compute` by the corresponding function
        #             if isinstance(other, Reward): # callable
        #                 def compute():
        #                     return op(self(), other())
        #             else:
        #                 def compute():
        #                     return op(self(), other)
        #             reward.compute = compute
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
        #             reward.compute = compute
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
        return self._state

    @state.setter
    def state(self, state):
        if state is not None and not isinstance(state, State):
            raise TypeError("Expecting state to be None or an instance of State.")
        self._state = state

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action):
        if action is not None and not isinstance(action, Action):
            raise TypeError("Expecting action to be None or an instance of Action.")
        self._action = action

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
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

    ###########
    # Methods #
    ###########

    def has_rewards(self):
        return len(self._rewards) > 0

    @staticmethod
    def is_maximized():
        return True

    def compute(self):
        pass

    #############
    # Operators #
    #############

    def __repr__(self):
        if not self.rewards or self.rewards is None:
            return self.__class__.__name__
        else:
            lst = [reward.__repr__() for reward in self.rewards]
            return ' + '.join(lst)

    def __call__(self, *args, **kwargs):
        return self.compute()

    # for unary and binary operators, see `__init__()` method.

    def __build_reward(self, other):
        # built the internal list of rewards
        rewards = self._rewards if self.has_rewards() else [self]
        if isinstance(other, Reward):
            if other.has_rewards():
                rewards.extend(other._rewards)
            else:
                rewards.append(other)
        return Reward(rewards=rewards)

    def __get_operation(self, other, op):
        if isinstance(other, Reward):  # callable
            def compute():
                return op(self(), other())
        else:
            def compute():
                return op(self(), other)
        return compute

    def __add__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__add__)
        return reward

    def __div__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__div__)
        return reward

    def __floordiv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__floordiv__)
        return reward

    def __iadd__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__iadd__)
        return reward

    def __idiv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__idiv__)
        return reward

    def __ifloordiv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__ifloordiv__)
        return reward

    def __imod__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__imod__)
        return reward

    def __imul__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__imul__)
        return reward

    def __ipow__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__ipow__)
        return reward

    def __isub__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__isub__)
        return reward

    def __itruediv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__itruediv__)
        return reward

    def __mod__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__mod__)
        return reward

    def __mul__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__mul__)
        return reward

    def __pow__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__pow__)
        return reward

    def __radd__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__add__)
        return reward

    def __rdiv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__div__)
        return reward

    def __rfloordiv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__floordiv__)
        return reward

    def __rmod__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__mod__)
        return reward

    def __rmul__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__mul__)
        return reward

    def __rpow__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__pow__)
        return reward

    def __rsub__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__sub__)
        return reward

    def __rtruediv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__truediv__)
        return reward

    def __sub__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__sub__)
        return reward

    def __truediv__(self, other):
        reward = self.__build_reward(other)
        reward.compute = self.__get_operation(other, operator.__truediv__)
        return reward

    # binary comparison operators
    def __eq__(self, other):
        compute = self.__get_operation(other, operator.__eq__)
        return compute()

    def __ge__(self, other):
        compute = self.__get_operation(other, operator.__ge__)
        return compute()

    def __gt__(self, other):
        compute = self.__get_operation(other, operator.__gt__)
        return compute()

    def __le__(self, other):
        compute = self.__get_operation(other, operator.__le__)
        return compute()

    def __lt__(self, other):
        compute = self.__get_operation(other, operator.__lt__)
        return compute()

    def __ne__(self, other):
        compute = self.__get_operation(other, operator.__ne__)
        return compute()

    # unary operators
    def __abs__(self):
        reward = copy.copy(self)  # shallow copy
        reward.compute = lambda: operator.__abs__(self())
        return reward

    def __neg__(self):
        reward = copy.copy(self)  # shallow copy
        reward.compute = lambda: operator.__neg__(self())
        return reward

    def __pos__(self):
        reward = copy.copy(self)  # shallow copy
        reward.compute = lambda: operator.__pos__(self())
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
#                 y.compute = f
#                 return y
#             else:
#                 return op(x)
#         return fct
#
#     globals()[name] = wrapper(getattr(np, name))

def ceil(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.ceil(x())
        return y
    else:
        return np.ceil(x)


def cos(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.cos(x())
        return y
    else:
        return np.cos(x)


def cosh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.cosh(x())
        return y
    else:
        return np.cosh(x)


def degrees(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.degrees(x())
        return y
    else:
        return np.degrees(x)


def exp(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.exp(x())
        return y
    else:
        return np.exp(x)


def expm1(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.expm1(x())
        return y
    else:
        return np.expm1(x)


def floor(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.floor(x())
        return y
    else:
        return np.floor(x)


def frexp(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.frexp(x())
        return y
    else:
        return np.frexp(x)


def isinf(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.isinf(x())
        return y
    else:
        return np.isinf(x)


def isnan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.isnan(x())
        return y
    else:
        return np.isnan(x)


def log(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.log(x())
        return y
    else:
        return np.log(x)


def log10(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.log10(x())
        return y
    else:
        return np.log10(x)


def log1p(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.log1p(x())
        return y
    else:
        return np.log1p(x)


def modf(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.modf(x())
        return y
    else:
        return np.modf(x)


def radians(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.radians(x())
        return y
    else:
        return np.radians(x)


def sin(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.sin(x())
        return y
    else:
        return np.sin(x)


def sinh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.sinh(x())
        return y
    else:
        return np.sinh(x)


def sqrt(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.sqrt(x())
        return y
    else:
        return np.sqrt(x)


def tan(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.tan(x())
        return y
    else:
        return np.tan(x)


def tanh(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.tanh(x())
        return y
    else:
        return np.tanh(x)


def trunc(x):
    if callable(x):
        y = copy.copy(x)  # shallow copy
        y.compute = lambda: np.trunc(x())
        return y
    else:
        return np.trunc(x)


##############################################################
#                         Rewards                            #
##############################################################

class FixedReward(Reward):
    r"""Fixed reward.

    This is a dummy class which always returns a fixed reward. This is fixed initially.
    """

    def __init__(self, value):
        super(FixedReward, self).__init__()
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting a number")
        self.value = value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.value))

    def compute(self):
        return self.value


class FunctionalReward(Reward):
    r"""Functional reward.

    This is a reward class which calls a given function/class to compute the reward.
    """
    def __init__(self, function):
        super(FunctionalReward, self).__init__()
        self.function = function

    def __repr__(self):
        return self.function.__name__

    def compute(self):
        return self.function()


class ForwardProgressReward(Reward):
    r"""Forward progress reward

    Compute the forward progress based on a forward direction, a previous and current positions.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=False):
        super(ForwardProgressReward, self).__init__(state=state)

        # if direction is None:
        #     # takes the robot initial direction
        #     #direction = ...
        #     #init_pos
        #     pass
        # if isinstance(direction, np.ndarray):
        #     pass
        # elif isinstance(direction, Robot):
        #     # takes the
        #     pass
        #
        # self.direction = direction
        # #self.init_pos = init_pos

        self.direction = self.normalize(np.array(direction))

        # TODO uncomment
        # if not isinstance(state, (PositionState, BasePositionState)):
        #     raise ValueError("Expecting state to be a PositionState or BasePositionState")
        self.init_pos = np.copy(self.state._data)
        self.value = 0

    @staticmethod
    def normalize(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def compute(self):
        curr_pos = self.state._data
        delta_pos = curr_pos - self.init_pos
        self.value = self.direction.dot(delta_pos)
        # self.value = curr_pos[0] - self.init_pos[0]
        self.init_pos = np.copy(curr_pos)
        return self.value


class DirectiveReward(Reward):
    r"""Directive Reward

    Provide reward if the vector state is in the specified direction. Specifically, it computes the dot product
    between the state vector and the specified direction.

    If normalize, the reward is between -1 and 1.
    """

    def __init__(self, state, direction=(1, 0, 0), normalize=True):
        super(DirectiveReward, self).__init__(state=state)

        self.normalize = normalize
        if self.normalize:
            self.direction = self.norm(np.array(direction))

        # TODO uncomment
        # if not isinstance(state, (PositionState, BasePositionState)):
        #     raise ValueError("Expecting state to be a PositionState or BasePositionState")
        self.value = 0

    @staticmethod
    def norm(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def compute(self):
        pos = self.state._data
        if self.normalize:
            pos = self.norm(pos)
        self.value = self.direction.dot(pos)
        return self.value


class L2SimilarityReward(Reward):
    """
    Compute the square of the L2 norm between two vectors.
    """

    def __init__(self):
        super(L2SimilarityReward, self).__init__()

    def value(self, vector1, vector2):
        return np.dot(vector1, vector2)


class ImitationReward(Reward):

    def __init__(self, human, robot):
        super(ImitationReward, self).__init__()
        self.human = human  # instance of HumanKinematic class
        self.robot = robot  # instance of Robot class

    def compute(self):
        # check
        pass


class GymReward(Reward):
    r"""OpenAI Gym reward

    This provides a wrapper
    """

    def __init__(self, value):
        super(GymReward, self).__init__()
        if not isinstance(value, (int, float)):
            raise TypeError("Expecting a number")
        self.value = value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.value))

    def compute(self):
        return self.value


# Test
if __name__ == '__main__':
    reward = 2*FixedReward(10) + FixedReward(3)**2 - 10
    reward += FixedReward(2)
    print(reward())
    print(isinstance(reward, Reward))
    print(reward.rewards)

    reward = FixedReward(-10)
    print(reward())
    reward = abs(reward)
    print(reward())
    print(FixedReward(10) == FixedReward(10))

    print('')
    reward = FixedReward(2) + FixedReward(1)
    print(reward())
    reward = cos(reward)
    print(reward())
    # print(type(reward))
    # print(reward.rewards)
