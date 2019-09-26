# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the joystick interface states.
"""

import numpy as np

from pyrobolearn.states.interfaces.interface_state import InterfaceState
import pyrobolearn.tools.interfaces.controllers as controllers


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JoystickState(InterfaceState):
    r"""Joystick Interface state.

    This is a state that reads the data from a joystick or game controller interface.
    """

    def __init__(self, interface):
        """
        Initialize the game controller (input) interface state.

        Args:
            interface (GameControllerInterface): game controller input interface.
        """
        if not isinstance(interface, controllers.GameControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `GameControllerInterface`, instead "
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


class XboxControllerState(JoystickState):
    r"""Xbox Controller state.
    """

    def __init__(self, interface):
        """
        Initialize the Xbox controller state.

        Args:
            interface (XboxControllerInterface): Xbox controller input interface.
        """
        if not isinstance(interface, controllers.xbox.XboxControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `XboxControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(XboxControllerState, self).__init__(interface)


class Xbox360ControllerState(XboxControllerState):
    r"""Xbox 360 controller state.
    """

    def __init__(self, interface=None, use_thread=False):
        """
        Initialize the Xbox 360 controller state.

        Args:
            interface (Xbox360ControllerInterface, None): Xbox 360 controller input interface. If None, it will
                initialize the interface.
            use_thread (bool): If True, it will run the interface in a thread. Otherwise, it will run the interface
                everytime it is called.
        """
        if interface is None:
            interface = controllers.xbox.Xbox360ControllerInterface(use_thread=use_thread)
        if not isinstance(interface, controllers.xbox.Xbox360ControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `Xbox360ControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(Xbox360ControllerState, self).__init__(interface)


class XboxOneControllerState(XboxControllerState):
    r"""Xbox 360 controller state.
    """

    def __init__(self, interface=None, use_thread=False):
        """
        Initialize the Xbox One controller state.

        Args:
            interface (XboxOneControllerInterface, None): Xbox 360 controller input interface. If None, it will
                initialize the interface.
            use_thread (bool): If True, it will run the interface in a thread. Otherwise, it will run the interface
                everytime it is called.
        """
        if interface is None:
            interface = controllers.xbox.XboxOneControllerInterface(use_thread=use_thread)
        if not isinstance(interface, controllers.xbox.XboxOneControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `XboxOneControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(XboxOneControllerState, self).__init__(interface)


class PSControllerState(JoystickState):
    r"""Playstation Controller State.
    """

    def __init__(self, interface):
        """
        Initialize the PS controller state.

        Args:
            interface (PSControllerInterface): Playstation controller input interface.
        """
        if not isinstance(interface, controllers.playstation.PSControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `PSControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(PSControllerState, self).__init__(interface)


class PS3ControllerState(PSControllerState):
    r"""PlayStation 3 controller state.
    """

    def __init__(self, interface=None, use_thread=False):
        """
        Initialize the PlayStation 3 controller state.

        Args:
            interface (PS3ControllerInterface, None): PlayStation 3 controller input interface. If None, it will
                initialize the interface.
            use_thread (bool): If True, it will run the interface in a thread. Otherwise, it will run the interface
                everytime it is called.
        """
        if interface is None:
            interface = controllers.playstation.PS3ControllerInterface(use_thread=use_thread)
        if not isinstance(interface, controllers.playstation.PS3ControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `PS3ControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(PS3ControllerState, self).__init__(interface)


class PS4ControllerState(PSControllerState):
    r"""PlayStation 4 controller state.
    """

    def __init__(self, interface=None, use_thread=False):
        """
        Initialize the PlayStation 4 controller state.

        Args:
            interface (PS4ControllerInterface, None): PlayStation 4 controller input interface. If None, it will
                initialize the interface.
            use_thread (bool): If True, it will run the interface in a thread. Otherwise, it will run the interface
                everytime it is called.
        """
        if interface is None:
            interface = controllers.playstation.PS4ControllerInterface(use_thread=use_thread)
        if not isinstance(interface, controllers.playstation.PS4ControllerInterface):
            raise TypeError("Expecting the given interface to be an instance of `PS3ControllerInterface`, instead "
                            "got: {}".format(type(interface)))
        super(PS4ControllerState, self).__init__(interface)
