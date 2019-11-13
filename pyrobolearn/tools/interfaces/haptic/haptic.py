#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the haptic interface.

Prerequisites:
- follow instructions in [4] on how to install h3dapi.
- To use the Python library, you have to run `H3DLoad <file>.x3d` where in the `<file>.x3d` you make a reference to
the Python script you want to run. Inside this Python script, you will be able to access to the Python library
`H3DInterface`... This is bad... Need to find a better alternative.

Troubleshooting:
- Check: https://www.h3dapi.org/modules/mediawiki/index.php/H3DAPI_FAQ#Python
-

Dependencies:
- `pyrobolearn.tools.interfaces`

References:
    - [1] Haptic devices from Force Dimension: http://www.forcedimension.com/
    - [2] Haptic devices from 3D Systems: https://www.3dsystems.com/haptics-devices/openhaptics
    - [3] H3D API: https://h3dapi.org/
    - [4] H3D API - Documentation: https://www.h3dapi.org/modules/mediawiki/index.php/Main_Page
    - [6] H3D API - Github repo: https://github.com/SenseGraphics/h3dapi
    - [7] Haptic devices: https://www.h3dapi.org/modules/mediawiki/index.php/Haptics_devices
"""

# TODO

try:
    import H3DInterface as h3d
except ImportError as e:
    string = "\nHint: try to install h3dapi by following the instructions on: " \
             "https://www.h3dapi.org/modules/mediawiki/index.php/H3DAPI_Installation" \
             "\nNote that you will have to install the API from source to have the Python library."
    raise ImportError(e.__str__() + string)

from pyrobolearn.tools.interfaces import InputOutputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class HapticInterface(InputOutputInterface):
    r"""Haptic Interface

    Provide the haptic interface which, for instance, allows the user to interact with the world and the robots.

    References:
        - [1] Haptic devices from Force Dimension: http://www.forcedimension.com/
        - [2] Haptic devices from 3D Systems: https://www.3dsystems.com/haptics-devices/openhaptics
        - [3] H3D API: https://h3dapi.org/
        - [4] H3D API - Documentation: https://www.h3dapi.org/modules/mediawiki/index.php/Main_Page
        - [6] H3D API - Github repo: https://github.com/SenseGraphics/h3dapi
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False):
        """
        Initialize the SpaceMouse interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """

        # From [7]
        # root, = references.getValue()
        #
        # di = getActiveDeviceInfo()
        # if di:
        #     hd = di.device.getValue()[0]

        # available fields
        # hd.trackerPosition
        # hd.trackerOrientation
        # hd.mainButton
        # hd.secondaryButton

        super(HapticInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    ###########
    # Methods #
    ###########

    def run(self):
        """Run the interface."""
        pass


# Test the interface
if __name__ == '__main__':
    pass
