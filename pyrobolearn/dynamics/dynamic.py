#!/usr/bin/env python
"""Provides the `transition`/`dynamic` function approximators in RL.

Dynamic models allows to compute the next state given the current state and action; that is, p(s_{t+1} | s_t, a_t).

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.actions`
- `pyrobolearn.approximators` (and thus `pyrobolearn.models`)
"""

from abc import ABCMeta, abstractmethod

from pyrobolearn.states import State
from pyrobolearn.actions import Action
# from pyrobolearn.approximators import Approximator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class DynamicModel(object):
    r"""Dynamic/Transition Model

    In the reinforcement learning setting, the dynamic model is the transition function associated to the environment
    which describes how ... The agent/policy has no control over it. However, it can often be learned from data samples
    acquired by interacting with the environment. This allows to model the environment and then perform internal
    simulations which are then more sample efficient in the real environment.

    When a dynamic model is involved, this is known as "Model-based Reinforcement Learning". These methods usually
    requires less samples as they learn the model of the environment.

    .. math::

        P_{\varphi}(s_{t+1} | s_t, a_t)

    They are 2 main ways to build a dynamic model:
    1. build it from a mathematical model
        Pros: mathematical guarantees (such as stability,...), predictable,...
        Cons: linearization, unmodeled phenomenon, assumptions that might be violated (rigid body), complex...
    2. learn it from the data, by letting the policy interacts with the environment
        Pros:
        Cons: usually requires a lot of samples to be accurate, mismatch between the real and the learned dynamic
              model, often no guarantees and could be unpredictable

    Note that learning a wrong dynamic model can have drastic consequences on the learned policy. Indeed, learning
    a dynamic model in the simulator can be completely to a learned .

    Some papers have worked on simulators that generates ...

    """
    __metaclass__ = ABCMeta

    def __init__(self, states, actions, model=None):
        self.states = self._check_states(states)
        self.actions = self._check_actions(actions)

    @staticmethod
    def _check_states(states):
        """
        Check if the states are valid (i.e. it is an instance of `State`, a list/tuple of `State` instances, or None).
        :param states: states to be checked.
        :return: states
        """
        if isinstance(states, State):
            states = [states]
        elif isinstance(states, (list, tuple)):
            for state in states:
                if not isinstance(state, State):
                    raise ValueError("Each state in the list/tuple must be a `State` object.")
        elif states is None:  # some policies don't need the state information
            states = []
        else:
            raise ValueError("The `states` parameter must be a `State` object or a list/tuple of `State` objects.")
        return states

    @staticmethod
    def _check_actions(actions):
        """
        Check if the actions are valid (i.e. it is an instance of `Action`, a list/tuple of `Action` instances).
        :param actions: actions to be checked.
        :return: actions
        """
        if isinstance(actions, Action):
            actions = [actions]
        elif isinstance(actions, (list, tuple)):
            for action in actions:
                if not isinstance(action, Action):
                    raise ValueError("Each action in the list/tuple must be an instance of the `Action` class.")
        else:
            raise ValueError("The `actions` parameter must be an `Action` object or a list/tuple of `Action` objects.")
        return actions

    @abstractmethod
    def __call__(self, states, actions):
        """
        Return predicted state given the current state and action
        :param state:
        :param action:
        :return:
        """
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class PhysicalDynamicModel(DynamicModel):
    r"""Physical Dynamic Model

    Dynamic model described by mathematical/physical equations.
    """
    def __init__(self, states, actions):
        super(PhysicalDynamicModel, self).__init__(states, actions)


class RobotDynamicModel(PhysicalDynamicModel):
    r"""Robot Dynamical Model

    This is the mathematical model of the robots.

    Limitations:
    * mathematical assumptions such as rigid bodies
    * the states/actions have to be robot states/actions
    """
    def __init__(self, states, actions):
        super(RobotDynamicModel, self).__init__(states, actions)


class LinearDynamicModel(DynamicModel):
    r"""Linear Dynamic Model

    Pros: easy to implement
    Cons: very limited
    """
    def __init__(self, states, actions):
        super(LinearDynamicModel, self).__init__(states, actions)


class PieceWiseLinearDynamicModel(DynamicModel):
    r"""Piecewise linear dynamic model

    Pros: easy to implement, often good predictions in local regions
    Cons: poor scalability
    """
    def __init__(self, states, actions):
        super(PieceWiseLinearDynamicModel, self).__init__(states, actions)


class NNDynamicModel(DynamicModel):
    r"""Neural Network Dynamic Model

    Dynamic model using neural networks.

    Pros:
    Cons: requires lot of samples, overfitting,...
    """
    def __init__(self, states, actions):
        super(NNDynamicModel, self).__init__(states, actions)


class GPDynamicModel(DynamicModel):
    r"""Gaussian Process Dynamic Model

    Dynamic model using Gaussian Processes.

    Pros: good from a mathematical point of view: integrate uncertainty on the dynamic model
    Cons:

    ..seealso: PILCO
    """
    def __init__(self, states, actions):
        super(GPDynamicModel, self).__init__(states, actions)
