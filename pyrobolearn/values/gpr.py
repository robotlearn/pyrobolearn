#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Gaussian Process Regression (GPR) value function approximators.

Define the GPR value that can be used.
"""

import numpy as np
import torch

from pyrobolearn.approximators.gp import GPRApproximator
from pyrobolearn.values.value import ParametrizedValue, ParametrizedQValue, ParametrizedQValueOutput


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GPRValue(ParametrizedValue):
    r"""GPR State Value Function Approximator

    State value function :math:`V_{\phi}(s)` approximated by a GPR model, where :math:`\phi` represents
    the parameters of that model.
    """

    def __init__(self, state, preprocessors=None):
        """
        Initialize the GPR state value function approximator.

        Args:
            state (State): input state.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = GPRApproximator(inputs=state, outputs=torch.Tensor([1]), preprocessors=preprocessors)
        super(GPRValue, self).__init__(state, model=model)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, preprocessors=self.model.preprocessors)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        state = copy.deepcopy(self.state, memo)
        preprocessors = [copy.deepcopy(preprocessor, memo) for preprocessor in self.model.preprocessors]
        value = self.__class__(state=state, preprocessors=preprocessors)
        memo[self] = value
        return value


class GPRQValue(ParametrizedQValue):
    r"""GPR Q-value function approximator (which accepts as inputs the states and actions)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a GPR model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and actions :math:`a`,
    and outputs the value :math:`Q(s,a)`. This can be used for continuous actions as well as discrete actions.
    """

    def __init__(self, state, action, preprocessors=None):
        """
        Initialize the GPR state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input action.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = GPRApproximator(inputs=[state, action], outputs=torch.Tensor([1]), preprocessors=preprocessors)
        super(GPRQValue, self).__init__(state, action, model=model)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action, preprocessors=self.model.preprocessors)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        state = copy.deepcopy(self.state, memo)
        action = copy.deepcopy(self.action, memo)
        preprocessors = [copy.deepcopy(preprocessor, memo) for preprocessor in self.model.preprocessors]
        value = self.__class__(state=state, action=action, preprocessors=preprocessors)
        memo[self] = value
        return value


class GPRQValueOutput(ParametrizedQValueOutput):
    r"""GPR Q-value function approximator (which accepts as inputs the states and outputs a Q-value for each
    discrete action)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a GPR model, where :math:`\phi` represents
    the parameters of that model. This approximator accepts as inputs the states :math:`s` and outputs the value
    :math:`Q(s,a)` for each discrete action. This can NOT be used with continuous actions.
    """

    def __init__(self, state, action, preprocessors=None):
        """
        Initialize the GPR state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output action.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = GPRApproximator(inputs=state, outputs=action, preprocessors=preprocessors)
        super(GPRQValueOutput, self).__init__(state, action, model=model)

    def __copy__(self):
        """Return a shallow copy of the value approximator. This can be overridden in the child class."""
        return self.__class__(state=self.state, action=self.action, preprocessors=self.model.preprocessors)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the value approximator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        state = copy.deepcopy(self.state, memo)
        action = copy.deepcopy(self.action, memo)
        preprocessors = [copy.deepcopy(preprocessor, memo) for preprocessor in self.model.preprocessors]
        value = self.__class__(state=state, action=action, preprocessors=preprocessors)
        memo[self] = value
        return value
