#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define various basic policies.

Define the various basic policies such as the random policy, linear policy, policies based on value functions, etc.
"""

import numpy as np
import torch

from pyrobolearn.policies.policy import Policy
from pyrobolearn.approximators import LinearApproximator
from pyrobolearn.values.value import ParametrizedQValueOutput


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RandomPolicy(Policy):
    """Random policy
    """

    class RandomModel(object):
        """Random model"""
        def __init__(self, actions, seed=None):
            self.seed = seed
            self.actions = actions

        @property
        def seed(self):
            return self._seed

        @seed.setter
        def seed(self, seed):
            if seed is not None:
                np.random.seed(seed)
            self._seed = seed

        def reset(self):
            pass

        def predict(self, state=None, to_numpy=True):
            spaces = self.actions.space
            return [space.sample() for space in spaces]

    def __init__(self, state, action, rate=1, seed=None, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the Random policy.

        Args:
            action (Action): At each step, by calling `policy.act(state)`, the `action` is computed by the policy,
                and can be given to the environment. As with the `state`, the type and size/shape of each inner
                action can be inferred and could be used to automatically build a policy. The `action` connects the
                policy with a controllable object (such as a robot) in the environment.
            state (State): By giving the `state` to the policy, it can automatically infer the type and size/shape
                of each inner state, and thus can be used to automatically build a policy. At each step, the `state`
                is filled by the environment, and read by the policy. The `state` connects the policy with one or
                several objects (including robots) in the environment. Note that some policies don't use any state
                information.
            model (Approximator, Model, None): inner model or approximator
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments
            **kwargs (dict): dictionary of arguments
        """
        model = self.RandomModel(action, seed=seed)
        super(RandomPolicy, self).__init__(state, action, model=model, rate=rate, preprocessors=preprocessors,
                                           postprocessors=postprocessors, *args, **kwargs)

    def sample(self, state=None):
        return self.act(state)


class LinearPolicy(Policy):
    """Linear Policy
    """

    def __init__(self, state, action, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the Linear Policy.

        Args:
            action (Action): At each step, by calling `policy.act(state)`, the `action` is computed by the policy,
                and can be given to the environment. As with the `state`, the type and size/shape of each inner
                action can be inferred and could be used to automatically build a policy. The `action` connects the
                policy with a controllable object (such as a robot) in the environment.
            state (State): By giving the `state` to the policy, it can automatically infer the type and size/shape
                of each inner state, and thus can be used to automatically build a policy. At each step, the `state`
                is filled by the environment, and read by the policy. The `state` connects the policy with one or
                several objects (including robots) in the environment. Note that some policies don't use any state
                information.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments
            **kwargs (dict): dictionary of arguments
        """
        model = LinearApproximator(state, action, preprocessors=preprocessors, postprocessors=postprocessors)
        super(LinearPolicy, self).__init__(state, action, model, rate=rate, *args, **kwargs)


class PolicyFromQValue(Policy):
    r"""Policy from Q-value function approximator

    This computes the optimal discrete action :math:`a` using the underlying value function approximator
    :math:`Q(s,a)` which given the state as input computes the Q-value for each discrete action. The policy select
    the best action by computing :math:`a = argmax_a Q_{\phi}(s,a)`.

    .. seealso::

        * `value.py`
    """

    def __init__(self, value, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the Policy from the value function approximator.

        Args:
            value (ParametrizedQValueOutput): trainable value function approximator.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments
            **kwargs (dict): dictionary of arguments
        """
        self.value = value
        super(PolicyFromQValue, self).__init__(state=value.state, action=value.action, model=value, rate=rate,
                                               preprocessors=preprocessors, postprocessors=postprocessors,
                                               *args, **kwargs)

    ##############
    # Properties #
    ##############

    @property
    def value(self):
        """Return the value function approximator."""
        return self._value

    @value.setter
    def value(self, value):
        """Set the value function approximator."""
        # TODO: need to check that input and output dimensions of the new value function approximator match the ones
        #  from the previous model.
        if not isinstance(value, ParametrizedQValueOutput):
            raise TypeError("Expecting the given `value` function approximator to be an instance of "
                            "`ParametrizedQValueOutput`, instead got: {}".format(type(value)))
        self._value = value

    ###########
    # Methods #
    ###########

    def inner_predict(self, state, deterministic=True, to_numpy=False, return_logits=True, set_output_data=False):
        """Inner prediction step.

        Args:
            state ((list of) torch.Tensor, (list of) np.array): state data.
            deterministic (bool): True by default. It can only be set to False, if the policy is stochastic.
            to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
            return_logits (bool): If True, in the case of discrete outputs, it will return the logits.
            set_output_data (bool): If True, it will set the predicted output data to the outputs given to the
                approximator.

        Returns:
            (list of) torch.Tensor, (list of) np.array: predicted action data.
        """
        values = self.value.evaluate(state, to_numpy=to_numpy)
        if return_logits:
            return values
        if to_numpy:
            return np.argmax(values)
        return torch.argmax(values, dim=0, keepdim=True)

    # def evaluate(self, state, to_numpy=False):
    #     """
    #     Evaluate the state.
    #
    #     Args:
    #         state (None, State, (list of) np.array, (list of) torch.Tensor): state input data. If None, it will get
    #             the data from the inputs that were given at the initialization.
    #         to_numpy (bool): If True, it will convert the data (torch.Tensors) to numpy arrays.
    #
    #     Returns:
    #         torch.Tensor, np.array:
    #     """
    #     values = self.value.evaluate(state, to_numpy=to_numpy)
    #     return values

    # def act(self, state=None, deterministic=True, to_numpy=True, return_logits=False, apply_action=True):
    #     pass

    # def sample(self, state):
    #     pass


# Tests
if __name__ == '__main__':
    import copy
    from pyrobolearn.states import FixedState
    from pyrobolearn.actions import FixedAction

    # check linear policy
    policy = LinearPolicy(state=FixedState(range(4)), action=FixedAction(range(2)))
    print(policy)

    target = copy.deepcopy(policy)
    print(target)
