#!/usr/bin/env python
"""Define the mouse keyboard interface states.
"""

import numpy as np

from pyrobolearn.states.interfaces.interface_state import InterfaceState
import pyrobolearn.tools.interfaces.mouse_keyboard as mouse_keyboard


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MouseKeyboardState(InterfaceState):
    r"""Mouse Keyboard Interface state.

    This is a state that reads the data from a mouse keyboard controller interface.
    """

    def __init__(self, interface):
        """
        Initialize the mouse keyboard (input) interface state.

        Args:
            interface (MouseKeyboardInterface): mouse keyboard input interface.
        """
        if not isinstance(interface, mouse_keyboard.MouseKeyboardInterface):
            raise TypeError("Expecting the given interface to be an instance of `MouseKeyboardInterface`, instead "
                            "got: {}".format(type(interface)))
        super(InterfaceState, self).__init__(interface)

    def _reset(self):
        """Reset the state and return the data."""
        pass

    def _read(self):
        """Read the next state or generate a state based on the :attr:`training_mode` that the state is in.
        If :attr:`interface.use_thread` is False and :attr:`training_mode` is True, then it has to update the
        interface as well.
        """
        pass
