#!/usr/bin/env python
"""Define basic actions

This includes notably the fixed and functional actions.
"""

import copy

from pyrobolearn.actions import Action


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class FixedAction(Action):
    r"""Fixed Action.

    This is a dummy fixed action which always returns the value it was initialized with.
    """

    def __init__(self, value):
        super(FixedAction, self).__init__(data=value)

    def _write(self, data=None):
        pass

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(value=self._data)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        data = copy.deepcopy(self._data)
        action = self.__class__(value=data)
        memo[self] = action
        return action


class FunctionalAction(Action):
    r"""Functional Action.

    This is an action which accepts a function which has to output the data.
    """

    def __init__(self, function, initial_data):
        self.function = function
        super(FunctionalAction, self).__init__(data=initial_data)

    def _write(self, data=None):
        self.data = self.function(data)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(function=self.function, data=self._data)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        function = copy.deepcopy(self.function)
        data = copy.deepcopy(self._data)
        action = self.__class__(function=function, initial_data=data)
        memo[self] = action
        return action
