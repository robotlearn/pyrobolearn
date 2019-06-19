#!/usr/bin/env python
"""Provide the PlayStation controller interfaces.

This provides the interfaces for the PlayStation controllers (PS3 and PS4) using the `inputs` library.
"""

try:
    from inputs import devices, get_gamepad
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


class PSControllerInterface(GameControllerInterface):
    r"""PlayStation Controller Interface

    This provides the code for the PlayStation Controller interface. We use the `inputs` Python library [1, 2].
    If the PS controller is not detected, please install the necessary drivers.

    References:
        [1] Python library `inputs`: https://inputs.readthedocs.io
        [2] Hardware support: https://inputs.readthedocs.io/en/latest/user/hardwaresupport.html
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False, controller_name='Sony'):
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

        if verbose:
            print(self.gamepad.name + ' detected.')

        if self.gamepad is None:
            raise ValueError("The specified gamepad/controller was not detected.")

        # translation
        buttons = ['BTN_SOUTH', 'BTN_EAST', 'BTN_WEST', 'BTN_NORTH', 'BTN_THUMBL', 'BTN_THUMBR', 'BTN_TL', 'BTN_TL2',
                   'BTN_TR', 'BTN_TR2', 'BTN_START', 'BTN_SELECT', 'BTN_MODE', 'ABS_HAT0X', 'ABS_HAT0Y', 'ABS_Z',
                   'ABS_RZ', 'ABS_X', 'ABS_Y', 'ABS_RX', 'ABS_RY']
        ps4_buttons = ['X', 'O', 'S', 'T', 'LJB', 'RJB', 'L1', 'L2', 'R1', 'R2', 'options', 'share', 'PS', 'L', 'R',
                       'RT', 'LJX', 'LJY', 'RJX', 'RJY']
        self.map = dict(zip(buttons, ps4_buttons))
        self.inv_map = dict(zip(ps4_buttons, buttons))

        # buttons and their values
        self.buttons = dict(zip(ps4_buttons[:12], [0] * 12))
        self.buttons.update(dict(zip(['Dpad', 'LJ', 'RJ'], [[0, 0]] * 3)))

        # last updated button
        self.last_updated_button = None

        super(PSControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    ##############
    # Properties #
    ##############

    @property
    def X(self):
        """Button X"""
        return self.buttons['X']

    @property
    def O(self):
        """Button O (circle)"""
        return self.buttons['O']

    # alias (C = circle)
    C = O

    @property
    def S(self):
        """Button Square"""
        return self.buttons['S']

    @property
    def T(self):
        """Button Triangle"""
        return self.buttons['T']

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
                self.buttons['LJ'][0] = value / 32768.  # values between [-32768, 32767]
                self.last_updated_button = 'LJ'
            elif key == 'LJY':
                self.buttons['LJ'][1] = -1. * value / 32768.  # values between [-32767, 32768]
                self.last_updated_button = 'LJ'
            elif key == 'RJX':
                self.buttons['RJ'][0] = value / 32768.  # values between [-32768, 32767]
                self.last_updated_button = 'RJ'
            elif key == 'RJY':
                self.buttons['RJ'][1] = -1. * value / 32768.  # values between [-32767, 32768]
                self.last_updated_button = 'RJ'
            elif key == 'DpadX':
                self.buttons['Dpad'][0] = value  # left (-1) and right (1)
                self.last_updated_button = 'Dpad'
            elif key == 'DpadY':
                self.buttons['Dpad'][1] = -1 * value  # down (-1) and high (1)
                self.last_updated_button = 'Dpad'
            elif key == 'LT' or key == 'RT':  # max 1023
                self.buttons[key] = value / 1023.
                # self.last_updated_button = key
        elif event_type == 'Key':
            print(event_type, key, value)
            self.buttons[key] = value
            self.last_updated_button = key


class PS3ControllerInterface(PSControllerInterface):
    r"""PlayStation 3 Controller Interface

    This provides the code for the PS3 Controller interface.

    In order to make this code works, make sure you installed the `inputs` Python library [1,2].
    If the PS4 controller is not detected, please install the necessary drivers.

    References:
        [1] Python library `inputs`: https://inputs.readthedocs.io
        [2] Hardware support: https://inputs.readthedocs.io/en/latest/user/hardwaresupport.html
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False,):
        super(PS3ControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose,
                                                     controller_name='Sony PLAYSTATION(R)3 Controller')


class PS4ControllerInterface(PSControllerInterface):
    r"""PlayStation 4 Controller Interface

    This provides the code for the PS4 Controller interface.

    In order to make this code works, make sure you installed the `inputs` Python library [1,2].
    If the PS4 controller is not detected, please install the necessary drivers.

    References:
        [1] Python library `inputs`: https://inputs.readthedocs.io
        [2] Hardware support: https://inputs.readthedocs.io/en/latest/user/hardwaresupport.html
    """

    def __init__(self, use_thread=False, sleep_dt=0, verbose=False,):
        super(PS4ControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose,
                                                     controller_name='Sony Interactive Entertainment Wireless '
                                                                     'Controller')


# Tests
if __name__ == '__main__':
    device = devices.gamepads[1]
    print(device.name)
    while True:
        events = device.read()  # blocking=False) # get_gamepad()
        for event in events:
            event_type, code, state = event.ev_type, event.code, event.state
            if event_type != 'Absolute':
                if code != 'SYN_REPORT':
                    print(code, state)
