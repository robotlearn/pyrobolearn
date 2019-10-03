# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the abstract interface state class.

The interface state allows to get the state information about input interfaces defined in
`pyrobolearn.tools.interfaces`.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.robots`
- `pyrobolearn.tools.interfaces.InputInterface`
"""

from pyrobolearn.states import State
from pyrobolearn.tools.interfaces.interface import InputInterface, InputOutputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class InterfaceState(State):
    r"""Input Interface state.

    This is a state that reads the data from an input interface (such as a mouse, keyboard, microphone, webcam /
    kinect, game controllers, VR/AR controllers, etc).
    """

    def __init__(self, interface):
        """
        Initialize the (input) interface state.

        Args:
            interface (InputInterface): input interface.
        """
        super(InterfaceState, self).__init__()

        # set the interface
        if not isinstance(interface, (InputInterface, InputOutputInterface)):
            raise TypeError("Expecting the given 'interface' to be an instance of `InputInterface` or "
                            "`InputOutputInterface`, instead got: {}".format(type(interface)))
        self._interface = interface

        # set training mode
        self._training_mode = False

    ##############
    # Properties #
    ##############

    @property
    def interface(self):
        """Return the input interface instance."""
        return self._interface

    @property
    def interface_in_thread(self):
        """Return True if the interface is running in another thread."""
        return self._interface.use_thread

    ###########
    # Methods #
    ###########

    def _reset(self):
        """Reset the state and return the data."""
        pass

    def _read(self):
        """Read the next state or generate a state based on the :attr:`training_mode` that the state is in.
        If :attr:`interface.use_thread` is False and :attr:`training_mode` is True, then it has to update the
        interface as well.
        """
        pass
