#!/usr/bin/env python
"""Provides the various basic value function approximators (e.g. table and linear value approximators)
"""

from abc import ABCMeta
import torch

# from pyrobolearn.models import Linear
from pyrobolearn.approximators import LinearApproximator
from pyrobolearn.values.value import ValueApproximator, ParametrizedValue, ParametrizedQValue, \
    ParametrizedQValueOutput


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ValueTable(ValueApproximator):
    r"""Value Table

    This is appropriate when the states/actions are discrete and have a low dimension.

    Dynamic Programming is used to compute the table.
    """

    def __init__(self, state):
        super(ValueTable, self).__init__(state)


class LinearValue(ParametrizedValue):
    r"""Linear State Value Function Approximator

    State value function :math:`V_{\phi}(s)` approximated by a linear model, where :math:`\phi` represents
    the parameters of that model.
    """

    def __init__(self, state, preprocessors=None):
        """
        Initialize the linear state value function approximator.

        Args:
            state (State): input state.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = LinearApproximator(inputs=state, outputs=torch.Tensor([1]), preprocessors=preprocessors)
        super(LinearValue, self).__init__(state, model=model)


class LinearQValue(ParametrizedQValue):
    r"""Linear Q-value function approximator (which accepts as inputs the states and actions)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a linear model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and actions :math:`a`,
    and outputs the value :math:`Q(s,a)`. This can be used for continuous actions as well as discrete actions.
    """

    def __init__(self, state, action, preprocessors=None):
        """
        Initialize the linear state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input action.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = LinearApproximator(inputs=[state, action], outputs=torch.Tensor([1]), preprocessors=preprocessors)
        super(LinearQValue, self).__init__(state, action, model=model)


class LinearQValueOutput(ParametrizedQValueOutput):
    r"""Linear Q-value function approximator (which accepts as inputs the states and outputs a Q-value for each
    discrete action)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a linear model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and outputs the value
    :math:`Q(s,a)` for each discrete action. This can NOT be used with continuous actions.
    """

    def __init__(self, state, action, preprocessors=None):
        """
        Initialize the linear state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output action.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = LinearApproximator(inputs=state, outputs=action, preprocessors=preprocessors)
        super(LinearQValueOutput, self).__init__(state, action, model=model)
