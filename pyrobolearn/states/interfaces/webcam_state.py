#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the webcam interface states.
"""

# TODO: debug

import numpy as np

from pyrobolearn.states.interfaces.interface_state import InterfaceState
import pyrobolearn.tools.interfaces.camera.webcam as webcam


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WebcamState(InterfaceState):
    r"""Webcam Interface state.

    This is a state that reads the data from a webcam or game controller interface.
    """

    def __init__(self, interface=None, use_thread=True, sleep_dt=1./10, verbose=False):
        """
        Initialize the webcam (input) interface state.

        Args:
            interface (WebcamInterface, None): webcam input interface. If None, it will initialize the interface.
            use_thread (bool): If True, it will run the interface in a thread. Otherwise, it will run the interface
                everytime it is called.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring the
                next sample.
            verbose (bool): If True, it will print information about the state of the webcam interface.
        """
        if interface is None:
            interface = webcam.WebcamInterface(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
        if not isinstance(interface, webcam.WebcamInterface):
            raise TypeError("Expecting the given interface to be an instance of `WebcamInterface`, instead "
                            "got: {}".format(type(interface)))
        super(WebcamState, self).__init__(interface)

    def _reset(self):
        """Reset the state and return the data."""
        self.data = self.interface.frame

    def _read(self):
        """Read the next state or generate a state based on the :attr:`training_mode` that the state is in.
        If :attr:`interface.use_thread` is False and :attr:`training_mode` is True, then it has to update the
        interface as well.
        """
        # if self.in_training_mode and not self.interface_in_thread
        self.data = self.interface.frame


# Tests
if __name__ == '__main__':
    from itertools import count
    import matplotlib.pyplot as plt

    # create webcam state
    state = WebcamState()

    # plotting using matplotlib in interactive mode
    fig = plt.figure()
    plot = None
    plt.ion()  # interactive mode on

    for _ in count():
        # # if don't use thread call `step` or `run` (note that `run` returns the frame but not
        # interface.step()

        # get the frame and plot it with matplotlib
        state()
        frame = state.data
        if len(frame) > 0:
            frame = frame[0]
            if plot is None:
                plot = plt.imshow(frame)
            else:
                plot.set_data(frame)
                plt.pause(0.01)

        # check if the figure is closed, and if so, get out of the loop
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()  # interactive mode off
    plt.show()
