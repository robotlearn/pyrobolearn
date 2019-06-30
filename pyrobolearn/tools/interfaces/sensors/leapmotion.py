#!/usr/bin/env python
"""Define the Leap Motion hand tracking sensor input interface.

References:
    - Leap Motion: https://www.leapmotion.com/
    - Installation (Ubuntu): https://www.leapmotion.com/setup/desktop/linux/
    - V2 tracking toolkit: https://developer.leapmotion.com/sdk/v2
    - Python API: https://developer-archive.leapmotion.com/documentation/python/api/Leap_Classes.html
    - Tutorials: https://www.youtube.com/playlist?list=PLgTGpidiW0iTELuljcIdTkA5SjHa5tudP
"""

# TODO: implement this interface

import numpy as np

try:
    import Leap
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `Leap` (see references in class documentation)')


from pyrobolearn.tools.interfaces.sensors import SensorInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LeapMotionInterface(SensorInterface):
    r"""Leap Motion Interface

    References:
        - Leap Motion: https://www.leapmotion.com/
        - Installation (Ubuntu): https://www.leapmotion.com/setup/desktop/linux/
        - V2 tracking toolkit: https://developer.leapmotion.com/sdk/v2
        - Python API: https://developer-archive.leapmotion.com/documentation/python/api/Leap_Classes.html
        - Tutorials: https://www.youtube.com/playlist?list=PLgTGpidiW0iTELuljcIdTkA5SjHa5tudP
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False, use_rgb=True, use_depth=True):
        """
        Initialize the RealSense input interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """

        # TODO

        super(LeapMotionInterface, self).__init__(use_thread, sleep_dt, verbose)

    ###########
    # Methods #
    ###########

    def run(self):
        """Run the interface."""
        pass  # TODO


# Test the interface
if __name__ == '__main__':
    pass
