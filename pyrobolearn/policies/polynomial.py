#!/usr/bin/env python
"""Provide the polynomial policy.

The polynomial policy uses a polynomial parametric approximator to predict the action vector based on the state vector.
"""

from pyrobolearn.policies.policy import Policy
from pyrobolearn.approximators import PolynomialApproximator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PolynomialPolicy(Policy):
    r"""Polynomial Policy

    The polynomial policy uses a polynomial parametric approximator: :math:`y = W \phi(x)` where :math:`x` is the state
    vector, and :math:`y` is the action vector, :math:`W` is the weight matrix, and :math:`\phi` is the polynomial
    function which returns the transformed state vector.
    """

    def __init__(self, state, action, degree=1, rate=1, preprocessors=None, postprocessors=None, *args, **kwargs):
        """
        Initialize the Polynomial Policy.

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
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            rate (int, float): rate (float) at which the policy operates if we are operating in real-time. If we are
                stepping deterministically in the simulator, it represents the number of ticks (int) to sleep before
                executing the model.
            preprocessors (Processor, list of Processor, None): pre-processors to be applied to the given input
            postprocessors (Processor, list of Processor, None): post-processors to be applied to the output
            *args (list): list of arguments
            **kwargs (dict): dictionary of arguments
        """
        model = PolynomialApproximator(state, action, degree=degree, preprocessors=preprocessors,
                                       postprocessors=postprocessors)
        super(PolynomialPolicy, self).__init__(state, action, model, rate=rate, *args, **kwargs)


# Tests
if __name__ == '__main__':
    import copy
    from pyrobolearn.states import FixedState
    from pyrobolearn.actions import FixedAction

    # check polynomial policy
    policy = PolynomialPolicy(state=FixedState(range(4)), action=FixedAction(range(2)))
    print(policy)

    target = copy.deepcopy(policy)
    print(target)
