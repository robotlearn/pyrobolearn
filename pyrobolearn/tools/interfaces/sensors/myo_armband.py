#!/usr/bin/env python
"""Define the Myo armband sensor input interface.

Warnings: "Myo production has officially ended as of Oct 12, 2018 and is no longer available for purchase".

References:
    - Myo Armband (webpage for support): https://support.getmyo.com/hc/en-us
    - PyoConnect: http://www.fernandocosentino.net/pyoconnect/
        - related useful repository: https://github.com/ijin/venus-lift-lights/blob/master/PyoConnect.py
    - MyoLinux: https://github.com/brokenpylons/MyoLinux
    - myo-python: https://github.com/NiklasRosenstein/myo-python
    - Software for Thalmic's Myo armband: https://github.com/balandinodidonato/MyoToolkit
"""

# TODO: implement this interface

import numpy as np

try:
    import myo
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `myo` or `PyoConnect`')


from pyrobolearn.tools.interfaces.sensors import SensorInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MyoArmbandInterface(SensorInterface):
    r"""Myo Armband Interface

    Warnings: "Myo production has officially ended as of Oct 12, 2018 and is no longer available for purchase".

    References:
        - Myo Armband (webpage for support): https://support.getmyo.com/hc/en-us
        - PyoConnect: http://www.fernandocosentino.net/pyoconnect/
            - related useful repository: https://github.com/ijin/venus-lift-lights/blob/master/PyoConnect.py
        - MyoLinux: https://github.com/brokenpylons/MyoLinux
        - myo-python: https://github.com/NiklasRosenstein/myo-python
        - Software for Thalmic's Myo armband: https://github.com/balandinodidonato/MyoToolkit
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

        super(MyoArmbandInterface, self).__init__(use_thread, sleep_dt, verbose)

    ###########
    # Methods #
    ###########

    def run(self):
        """Run the interface."""
        pass  # TODO


# Test the interface
if __name__ == '__main__':
    pass
