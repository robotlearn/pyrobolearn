#!/usr/bin/env python
"""Define the Myo armband sensor input interface.

Warnings: "Myo production has officially ended as of Oct 12, 2018 and is no longer available for purchase".

The Myo Armband has 8 EMG sensors (the sensor with the Myo logo is the 4th sensor, see picture on the web), and
an IMU sensor (which allows to get the orientation, acceleration from the accelerometer, and angular velocity
from the gyroscope).

The coordinate system is a right-handed coordinate system, and it is the same as the one used in robotics.
Once the armband is put on your arm and your elbow is bent at 90 degrees, the x-axis is pointing forward,
the y-axis is pointing to the left, and the z-axis is pointing upward. More information can be found on the web.

This implementation used a slightly modified ``Myo`` class from ``PyoConnect`` from [2], which inherits from
``MyoRaw`` from [3].

References:
    - [1] Myo Armband (webpage for support): https://support.getmyo.com/hc/en-us
    - [2] PyoConnect (Python + Linux): http://www.fernandocosentino.net/pyoconnect/
    - [3] myo-raw (Python + Linux): https://github.com/dzhu/myo-raw
"""

import numpy as np

try:
    from myo.PyoConnectLib import Myo
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `PyoConnect` (version 2), see the `install_myo_ubuntu.txt` file.')


from pyrobolearn.tools.interfaces.sensors import SensorInterface
from pyrobolearn.utils.transformation import get_quaternion_from_rpy


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

    The Myo Armband has 8 EMG sensors (the sensor with the Myo logo is the 4th sensor, see picture on the web), and
    an IMU sensor (which allows to get the orientation, acceleration from the accelerometer, and angular velocity
    from the gyroscope).

    The coordinate system is a right-handed coordinate system, and it is the same as the one used in robotics.
    Once the armband is put on your arm and your elbow is bent at 90 degrees, the x-axis is pointing forward,
    the y-axis is pointing to the left, and the z-axis is pointing upward. More information can be found on the web.

    This implementation used a slightly modified ``Myo`` class from ``PyoConnect`` from [2], which inherits from
    ``MyoRaw`` from [3].

    References:
        - [1] Myo Armband (webpage for support): https://support.getmyo.com/hc/en-us
        - [2] PyoConnect (Python + Linux): http://www.fernandocosentino.net/pyoconnect/
        - [3] myo-raw (Python + Linux): https://github.com/dzhu/myo-raw
    """

    def __init__(self, tty=None, use_thread=False, sleep_dt=0., verbose=False, library_verbose=False):
        """
        Initialize the Myo armband input interface.

        Args:
            tty (None, str): TTY (on Linux). You can check the list on a terminal by typing `ls /dev/tty*`. By default,
                it will be '/dev/ttyACM0'.
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
            library_verbose (bool): If True, it will print information returned by the library.
        """

        # create myo instance
        self.myo = Myo(tty=tty, verbose=library_verbose)

        # connect to myo (this is a blocking call)
        self.myo.connect()

        if verbose:
            print("Myo Armband is connected!")

        super(MyoArmbandInterface, self).__init__(use_thread, sleep_dt, verbose)

    ##############
    # Properties #
    ##############

    @property
    def roll(self):
        """Return the roll angle."""
        return self.myo.current_roll

    @property
    def pitch(self):
        """Return the pitch angle."""
        return self.myo.current_pitch

    @property
    def yaw(self):
        """Return the yaw angle."""
        return self.myo.current_yaw

    @property
    def rpy(self):
        """Return the roll-pitch-yaw angles."""
        return np.array([self.myo.current_roll, self.myo.current_pitch, self.myo.current_yaw])

    @property
    def quaternion(self):
        """Return the orientation expressed as a quaternion [x,y,z,w]."""
        return get_quaternion_from_rpy(self.rpy)

    @property
    def acceleration(self):
        """Return the values read by the accelerometer."""
        return np.asarray(self.myo.current_accel)

    @property
    def gyro(self):
        """Return the values read by the gyroscope."""
        return np.asarray(self.myo.current_gyro)

    @property
    def emg(self):
        """Return the EMG values returned by the 8 sensors."""
        return np.asarray(self.myo.current_emg_values)

    @property
    def current_arm(self):
        """Return a string 'left', 'right', or 'unknown' to specify which arm is attached the sensor."""
        return self.myo.current_arm

    @property
    def pose(self):
        """Return a string describing the last pose (between 'rest', 'fist', 'waveIn', 'waveOut', 'fingersSpread',
        'doubleTap', 'unknown')."""
        return self.myo.pose

    @property
    def x_direction(self):
        """Return a string describing the x direction (between 'unknown', 'towardWrist', and 'towardElbow')."""
        return self.myo.current_xdir

    ###########
    # Methods #
    ###########

    def vibrate(self, length):
        """Make the myo armband vibrate for length = [1,4]."""
        self.myo.vibrate(length)

    def run(self):
        """Run the interface."""
        ret = self.myo.run()
        if ret is None and self.verbose:
            print("Connection lost, trying to reconnect.")
            self.myo.connect()

    def close(self):
        """Close the interface."""
        self.myo.disconnect()

        if self.use_thread:
            if self.verbose:
                print("Asking for thread to stop...")
            self.stop_thread = True


# Test the interface
if __name__ == '__main__':
    import time

    # create the myo interface
    myo = MyoArmbandInterface(verbose=True)

    try:
        while True:
            myo.step()
            print("RPY: {}".format(myo.rpy))
            print("Quaternion: {}".format(myo.quaternion))
            print("EMG: {}".format(myo.emg))
            print("Accel: {}".format(myo.acceleration))
            print("Gyro: {}".format(myo.gyro))
            print("")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        myo.close()
        print("Bye!")
