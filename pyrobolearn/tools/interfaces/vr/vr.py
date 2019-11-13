#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the main Virtual Reality (VR) interface class.

References:
    - steamvr: https://www.steamvr.com/en/
    - SteamVR on Linux:
        - https://github.com/ValveSoftware/SteamVR-for-Linux
        - https://www.cgl.ucsf.edu/chimera/data/linux-vr-oct2018/ubuntuvr.html
    - OpenVR (not open source):
        - Github repository: https://github.com/ValveSoftware/openvr
        - Documentation: https://github.com/ValveSoftware/openvr/wiki/API-Documentation
    - PyOpenVR:
        - Github repository: https://github.com/cmbruns/pyopenvr
    - OSVR
        - http://www.osvr.org/
        - https://osvr.github.io/doc/installing/linux/
    - Misc:
        - How to use the HTC Vive Trackers in Ubuntu using Python 3.6: https://gist.github.com/DanielArnett/c9a56c9c7cc0def20648480bca1f6772
        - https://github.com/osudrl/CassieVrControls/wiki/Tracking:-OpenVR-SDK

Log:
$ sudo add-apt-repository multiverse
$ sudo apt install steam
$ sudo apt install steam-devices

* installing OSVR
"""

# TODO

# import openvr

from pyrobolearn.tools.interfaces.interface import InputOutputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class VRInterface(InputOutputInterface):
    r"""Virtual Reality Interface

    Virtual reality interface from which other VR interfaces (such as HTC, Oculus) inherit from.
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False):
        super(VRInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)
