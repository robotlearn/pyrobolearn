#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the RealSense depth camera input interface.

References:
    - librealsense: https://github.com/IntelRealSense/librealsense
    - python wrapper (pyrealsense2): https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
"""

# TODO: implement this interface

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `pyrealsens2`: pip install pyrealsense2'
                                '\nOr try to install from its repository: '
                                'https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python\n')


from pyrobolearn.tools.interfaces.camera import CameraInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RealSenseInterface(CameraInterface):
    r"""RealSense Interface

    References:
        - librealsense: https://github.com/IntelRealSense/librealsense
        - python wrapper (pyrealsense2): https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
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
            use_rgb (bool): If True, it will get the RGB camera image. Note that if this is enabled, you can not
                 get IR images at the same time.
            use_depth (bool): If True, it will get the depth camera image.
        """

        # TODO

        super(RealSenseInterface, self).__init__(use_thread, sleep_dt, verbose)

    ###########
    # Methods #
    ###########

    def run(self):
        """Run the interface."""
        pass  # TODO


# Test the interface
if __name__ == '__main__':
    pass
