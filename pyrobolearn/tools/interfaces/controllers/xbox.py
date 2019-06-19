#!/usr/bin/env python
"""Provide the Xbox controller interfaces.

This provides the interfaces for the Xbox controllers (Xbox 360 and Xbox One) using the `inputs` library.

Troubleshooting:
If the Xbox controller is not detected, please install the necessary driver. On Ubuntu 16.04, you can install the
`xpad` driver by typing the following commands (see [1]):

```bash
sudo apt-get install git
sudo apt-get install dkms
sudo git clone https://github.com/paroj/xpad.git /usr/src/xpad-0.4
sudo dkms install -m xpad -v 0.4
```

References:
    [1] https://askubuntu.com/questions/783587/how-do-i-get-an-xbox-one-controller-to-work-with-16-04-not-steam
"""

try:
    from inputs import devices
    # TODO: update the library inputs to make it non-blocking
    # solution: use the fcntl to read in a non-blocking mode, change `InputDevice._get_data(self, read_size)`,
    # instead of using `read`, use the fcntl library
    # solution1: use threads
    # References:
    # [1] https://github.com/zeth/inputs/pull/9/commits/e1356b945c8f47667fe2f0b4f13b9e8e7b83238a  (doesn't work)
    # [2] https://github.com/rakshit97/usb-controller-as-mouse  (gives a pointer)
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install `inputs`: pip install inputs')

from pyrobolearn.tools.interfaces.controllers.controller import GameControllerInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class XboxControllerInterface(GameControllerInterface):
    r"""Xbox Controller Interface.

    This provides the code for the Xbox Controller interface.

    In order to make this code works, make sure you installed the `inputs` Python library [1]. If the Xbox controller
    is not detected, please install the necessary driver. On Ubuntu 16.04, you can install the `xpad` driver by typing
    the following commands (see [3]):

    ```bash
    sudo apt-get install git
    sudo apt-get install dkms
    sudo git clone https://github.com/paroj/xpad.git /usr/src/xpad-0.4
    sudo dkms install -m xpad -v 0.4
    ```

    Notes: I tried the `xboxdrv` driver (https://github.com/xboxdrv/xboxdrv) on Ubuntu 16.04 (kernel 4.1* and 4.4),
    which was necessary for the following code repos:
    * https://github.com/FRC4564/Xbox
    * https://github.com/linusg/xbox360controller  (Note that this requires at least Python 3.3)
    but it didn't work. For more info, check [3]

    References:
        [1] Python library `inputs`: https://inputs.readthedocs.io
        [2] Hardware support: https://inputs.readthedocs.io/en/latest/user/hardwaresupport.html
        [3] https://askubuntu.com/questions/783587/how-do-i-get-an-xbox-one-controller-to-work-with-16-04-not-steam
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False, controller_name='X-Box One'):
        # Check if some gamepads are connected to the computer
        gamepads = devices.gamepads
        if len(gamepads) == 0:
            raise ValueError("No gamepads/controllers were detected.")

        # Check if the specified gamepad is detected
        self.gamepad = None
        for gamepad in gamepads:
            if controller_name in gamepad.name:
                self.gamepad = gamepad
                break

        if self.gamepad is None:
            raise ValueError("The specified gamepad/controller was not detected.")

        if verbose:
            print(self.gamepad.name + ' detected.')

        # translation
        buttons = ['BTN_SOUTH', 'BTN_EAST', 'BTN_WEST', 'BTN_NORTH', 'BTN_THUMBL', 'BTN_THUMBR', 'BTN_TL', 'BTN_TR',
                   'BTN_START', 'BTN_SELECT', 'ABS_Z', 'ABS_RZ', 'ABS_HAT0X', 'ABS_HAT0Y', 'ABS_X', 'ABS_Y', 'ABS_RX',
                   'ABS_RY']
        xbox_buttons = ['A', 'B', 'Y', 'X', 'LJB', 'RJB', 'LB', 'RB', 'menu', 'view', 'LT', 'RT', 'DpadX', 'DpadY',
                        'LJX', 'LJY', 'RJX', 'RJY']
        self.map = dict(zip(buttons, xbox_buttons))

        # buttons and their values
        self.buttons = dict(zip(xbox_buttons[:12], [0]*12))
        self.buttons.update(dict(zip(['Dpad', 'LJ', 'RJ'], [[0,0]]*3)))

        # last updated button
        self.last_updated_button = None

        super(XboxControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    ##############
    # Properties #
    ##############

    @property
    def A(self):
        """Button A"""
        return self.buttons['A']

    @property
    def B(self):
        """Button B"""
        return self.buttons['B']

    @property
    def Y(self):
        """Button Y"""
        return self.buttons['Y']

    @property
    def X(self):
        """Button X"""
        return self.buttons['X']

    @property
    def LJB(self):
        """Left Joystick Button"""
        return self.buttons['LJB']

    @property
    def RJB(self):
        """Right Joystick Button"""
        return self.buttons['RJB']

    @property
    def LB(self):
        """left bumper; button for left index finger"""
        return self.buttons['LB']

    @property
    def RB(self):
        """right bumper; button for right index finger"""
        return self.buttons['RB']

    @property
    def menu(self):
        """menu button"""
        return self.buttons['menu']

    @property
    def view(self):
        """view button"""
        return self.buttons['view']

    @property
    def LT(self):
        """Left trigger; button for left middle finger"""
        return self.buttons['LT']

    @property
    def RT(self):
        """Right trigger; button for right middle finger"""
        return self.buttons['RT']

    @property
    def Dpad(self):
        """Directional pad"""
        return self.buttons['Dpad']

    @property
    def LJ(self):
        """Left Joystick"""
        return self.buttons['LJ']

    @property
    def RJ(self):
        """Right Joystick"""
        return self.buttons['RJ']

    # aliases
    left_joystick = LJ
    right_joystick = RJ

    ###########
    # Methods #
    ###########

    def run(self):
        # print('running')
        events = self.gamepad.read()  # blocking=False) # get_gamepad()
        for event in events:
            event_type, code, state = event.ev_type, event.code, event.state
            self.__setitem(event_type, self.map.get(code), state)

            # display info
            if self.verbose and self.last_updated_button is not None:
                print("Pushed button {} - state = {}".format(self.last_updated_button,
                                                             self.buttons[self.last_updated_button]))

    def set_left_vibration(self, time_msec):
        """Set the vibration for the left motor"""
        self.gamepad.set_vibration(1, 0, time_msec)
        # display info
        if self.verbose:
            print("Set vibration to the left motor for {} msec".format(time_msec))

    def set_right_vibration(self, time_msec):
        """Set the vibration for the right motor"""
        self.gamepad.set_vibration(0, 1, time_msec)
        # display info
        if self.verbose:
            print("Set vibration to the right motor for {} msec".format(time_msec))

    def set_vibration(self, time_msec):
        """Set the vibration for both motors"""
        self.gamepad.set_vibration(1, 1, time_msec)
        # display info
        if self.verbose:
            print("Set vibration to both motors for {} msec".format(time_msec))

    def __getitem__(self, key):
        """Return the specified button"""
        if key is not None:
            return self.buttons[key]

    def __setitem(self, event_type, key, value):

        if event_type == 'Absolute':
            if key == 'LJX':
                self.buttons['LJ'][0] = value / 32768.          # values between [-32768, 32767]
                self.last_updated_button = 'LJ'
            elif key == 'LJY':
                self.buttons['LJ'][1] = -1. * value / 32768.    # values between [-32767, 32768]
                self.last_updated_button = 'LJ'
            elif key == 'RJX':
                self.buttons['RJ'][0] = value / 32768.          # values between [-32768, 32767]
                self.last_updated_button = 'RJ'
            elif key == 'RJY':
                self.buttons['RJ'][1] = -1. * value / 32768.    # values between [-32767, 32768]
                self.last_updated_button = 'RJ'
            elif key == 'DpadX':
                self.buttons['Dpad'][0] = value         # left (-1) and right (1)
                self.last_updated_button = 'Dpad'
            elif key == 'DpadY':
                self.buttons['Dpad'][1] = -1 * value    # down (-1) and high (1)
                self.last_updated_button = 'Dpad'
            elif key == 'LT' or key == 'RT':  # max 1023
                self.buttons[key] = value / 1023.
                self.last_updated_button = key
        elif event_type == 'Key':
            self.buttons[key] = value
            self.last_updated_button = key


class Xbox360ControllerInterface(XboxControllerInterface):
    r"""Xbox 360 Controller Interface

    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        super(Xbox360ControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt,
                                                         verbose=verbose, controller_name='X-Box 360')


class XboxOneControllerInterface(XboxControllerInterface):
    r"""Xbox One Controller Interface

    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False):
        super(XboxOneControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt,
                                                         verbose=verbose, controller_name='X-Box One')


# Tests
if __name__ == '__main__':
    import time
    from itertools import count

    # create interface
    xbox = XboxOneControllerInterface()

    for _ in count():
        # run one step with the interface
        xbox.run()  # same as `step()` if we are not using threads

        # get the last update and print it
        b = xbox.last_updated_button
        print("Last updated button: {} with value: {}".format(b, xbox[b]))

        # sleep a bit
        time.sleep(0.01)
