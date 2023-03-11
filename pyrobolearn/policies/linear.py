#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the linear policy.

The linear policy uses a linear parametric approximator to predict the action vector based on the state vector.
"""

from pyrobolearn.policies.policy import Policy
from pyrobolearn.approximators import LinearApproximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearPolicy(Policy):
    r"""Linear Policy

    The linear policy uses a linear parametric approximator: :math:`y = W x + b` where :math:`x` is the state vector,
    and :math:`y` is the action vector, :math:`W` is the weight matrix, and :math:`b` is the bias/intercept.
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
