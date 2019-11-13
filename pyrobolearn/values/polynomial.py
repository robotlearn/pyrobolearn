#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the polynomial value function approximator.
"""

import copy
import torch

from pyrobolearn.approximators import PolynomialApproximator
from pyrobolearn.values.value import ParametrizedValue, ParametrizedQValue, ParametrizedQValueOutput


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PolynomialValue(ParametrizedValue):
    r"""Polynomial State Value Function Approximator

    State value function :math:`V_{\phi}(s)` approximated by a polynomial model, where :math:`\phi` represents
    the parameters of that model.
    """

    def __init__(self, state, degree=1, preprocessors=None):
        """
        Initialize the polynomial state value function approximator.

        Args:
            state (State): input state.
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = PolynomialApproximator(inputs=state, outputs=torch.Tensor([1]), degree=degree,
                                       preprocessors=preprocessors)
        super(PolynomialValue, self).__init__(state, model=model)

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


class PolynomialQValue(ParametrizedQValue):
    r"""Polynomial Q-value function approximator (which accepts as inputs the states and actions)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a polynomial model, where :math:`\phi`
    represents the parameters of that model. This approximator accepts as inputs the states :math:`s` and actions
    :math:`a`, and outputs the value :math:`Q(s,a)`. This can be used for continuous actions as well as discrete actions.
    """

    def __init__(self, state, action, degree=1, preprocessors=None):
        """
        Initialize the polynomial state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): input action.
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = PolynomialApproximator(inputs=[state, action], outputs=torch.Tensor([1]), degree=degree,
                                       preprocessors=preprocessors)
        super(PolynomialQValue, self).__init__(state, action, model=model)

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


class PolynomialQValueOutput(ParametrizedQValueOutput):
    r"""Polynomial Q-value function approximator (which accepts as inputs the states and outputs a Q-value for each
    discrete action)

    State-action value function :math:`Q_{\phi}(s, a)` approximated by a polynomial model, where :math:`\phi`
    represents the parameters of that model. This approximator accepts as inputs the states :math:`s` and outputs the
    value :math:`Q(s,a)` for each discrete action. This can NOT be used with continuous actions.
    """

    def __init__(self, state, action, degree=1, preprocessors=None):
        """
        Initialize the polynomial state-action value function approximator.

        Args:
            state (State): input state.
            action (Action): output action.
            degree (int, list of int, np.array[D]): degree(s) of the polynomial. Setting `degree=3`, will apply
                `[1,x,x^2,x^3]` to the inputs, while setting `degree=[1,3]` will apply `[x,x^3]` to the inputs.
            preprocessors ((list of) Processor): pre-processors to be applied on the input state before being fed to
                the inner model / function approximator.
        """
        model = PolynomialApproximator(inputs=state, outputs=action, degree=degree, preprocessors=preprocessors)
        super(PolynomialQValueOutput, self).__init__(state, action, model=model)

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
