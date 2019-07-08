#!/usr/bin/env python
"""Provide the PlayStation controller interfaces.

This provides the interfaces for the PlayStation controllers (PS3 and PS4) using the `inputs` library.
"""

import numpy as np

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
        buttons = ['BTN_EAST', 'BTN_C', 'BTN_SOUTH', 'BTN_NORTH', 'BTN_SELECT', 'BTN_START', 'BTN_WEST', 'BTN_TL',
                   'BTN_Z', 'BTN_TR', 'BTN_TR2', 'BTN_TL2', 'BTN_MODE', 'BTN_THUMBL', 'ABS_HAT0X', 'ABS_HAT0Y',
                   'ABS_X', 'ABS_Y', 'ABS_Z', 'ABS_RZ']  # , 'ABS_RX', 'ABS_RY', 'BTN_THUMBR']
        ps4_buttons = ['X', 'O', 'S', 'T', 'LJB', 'RJB', 'L1', 'L2', 'R1', 'R2', 'options', 'share', 'PS', 'pad',
                       'DpadX', 'DpadY', 'LJX', 'LJY', 'RJX', 'RJY']
        self.map = dict(zip(buttons, ps4_buttons))
        self.inv_map = dict(zip(ps4_buttons, buttons))

        # buttons and their values
        self.buttons = dict(zip(ps4_buttons[:14], [0] * 14))
        self.buttons.update(dict(zip(['Dpad', 'LJ', 'RJ'], [np.array([0., 0.])] * 3)))

        # last updated button
        self.last_updated_button = None

        # pushed buttons

        super(PSControllerInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    ##############
    # Properties #
    ##############

    @property
    def X(self):
        """Button X"""
        return self.buttons['X']

    # alias
    cross = X

    @property
    def O(self):
        """Button O (circle)"""
        return self.buttons['O']

    # alias
    circle = O

    @property
    def S(self):
        """Button Square"""
        return self.buttons['S']

    # alias
    square = S

    @property
    def T(self):
        """Button Triangle"""
        return self.buttons['T']

    # alias
    triangle = T

    @property
    def LJB(self):
        """Left Joystick Button"""
        return self.buttons['LJB']

    @property
    def RJB(self):
        """Right Joystick Button"""
        return self.buttons['RJB']

    @property
    def L1(self):
        """left bumper; button for left index finger"""
        return self.buttons['L1']

    @property
    def R1(self):
        """right bumper; button for right index finger"""
        return self.buttons['R1']

    @property
    def L2(self):
        """Left trigger; button for left middle finger"""
        return self.buttons['L2']

    @property
    def R2(self):
        """Right trigger; button for right middle finger"""
        return self.buttons['R2']

    @property
    def options(self):
        """menu button"""
        return self.buttons['options']

    @property
    def share(self):
        """share button"""
        return self.buttons['share']

    @property
    def PS(self):
        """PS button"""
        return self.buttons['PS']

    @property
    def pad(self):
        """pad button"""
        return self.buttons['pad']

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

    # NOTE: the following buttons have been manually remapped to better correspond to what their name suggests
    @property
    def BTN_SOUTH(self):
        """South button"""
        return self.buttons['X']

    @property
    def BTN_EAST(self):
        """East button"""
        return self.buttons['O']

    @property
    def BTN_WEST(self):
        """West button"""
        return self.buttons['S']

    @property
    def BTN_NORTH(self):
        """North button"""
        return self.buttons['T']

    @property
    def BTN_C(self):
        """Circle button"""
        return self.buttons['O']

    @property
    def BTN_THUMBL(self):
        """Left thumb button"""
        return self.buttons['LJB']

    @property
    def BTN_THUMBR(self):
        """Right thumb button"""
        return self.buttons['RJB']

    @property
    def BTN_TL(self):
        """Left bumper; button for left index finger"""
        return self.buttons['L1']

    @property
    def BTN_TL2(self):
        """Left bumper 2; button for left middle finger"""
        return self.buttons['L2']

    @property
    def BTN_TR(self):
        """Right bumper; button for right index finger"""
        return self.buttons['R1']

    @property
    def BTN_TR2(self):
        """Right bumper 2; button for right middle finger"""
        return self.buttons['R2']

    @property
    def BTN_START(self):
        """Start button"""
        return self.buttons['share']

    @property
    def BTN_SELECT(self):
        """Select button"""
        return self.buttons['options']

    @property
    def BTN_MODE(self):
        """Mode button"""
        return self.buttons['PS']

    @property
    def ABS_Z(self):
        """Left trigger; non-existent for PS controller; return the same as L2."""
        return self.buttons['L2']

    @property
    def ABS_RZ(self):
        """Right trigger; non-existent for PS controller; return the same as R2."""
        return self.buttons['R2']

    @property
    def ABS_HAT0X(self):
        """Directional pad X position"""
        return self.buttons['Dpad'][0]

    @property
    def ABS_HAT0Y(self):
        """Directional pad Y position"""
        return self.buttons['Dpad'][1]

    @property
    def ABS_X(self):
        """Left joystick X position"""
        return self.buttons['LJ'][0]

    @property
    def ABS_Y(self):
        """Left joystick Y position"""
        return self.buttons['LJ'][1]

    @property
    def ABS_RX(self):
        """Right joystick X position"""
        return self.buttons['RJ'][0]

    @property
    def ABS_RY(self):
        """Right joystick Y position"""
        return self.buttons['RJ'][1]

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
                self.buttons['LJ'][0] = (value - 127.5) / 127.5  # values between [0, 255]
                self.last_updated_button = 'LJ'
            elif key == 'LJY':
                self.buttons['LJ'][1] = -1. * (value - 127.5) / 127.5  # values between [0, 255]
                self.last_updated_button = 'LJ'
            elif key == 'RJX':
                self.buttons['RJ'][0] = (value - 127.5) / 127.5  # values between [0, 255]
                self.last_updated_button = 'RJ'
            elif key == 'RJY':
                self.buttons['RJ'][1] = -1. * (value - 127.5) / 127.5  # values between [0, 255]
                self.last_updated_button = 'RJ'
            elif key == 'DpadX':
                self.buttons['Dpad'][0] = value  # left (-1) and right (1)
                self.last_updated_button = 'Dpad'
            elif key == 'DpadY':
                self.buttons['Dpad'][1] = -1 * value  # down (-1) and high (1)
                self.last_updated_button = 'Dpad'
            # elif key == 'LT' or key == 'RT':  # max 1023
            #     self.buttons[key] = value / 1023.
            #     # self.last_updated_button = key
        elif event_type == 'Key':
            # print(event_type, key, value)
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
    # create controller
    controller = PSControllerInterface(use_thread=True)
    print(controller.map)
    print(controller.buttons)

    # check buttons
    while True:
        # controller.step()
        if controller.cross:
            print("X button has been pushed.")
        if controller.square:
            print("Square button has been pushed.")
        if controller.circle:
            print("Circle button has been pushed.")
        if controller.triangle:
            print("Triangle button has been pushed.")
        if controller.L1:
            print("L1 has been pushed.")
        if controller.L2:
            print("L2 has been pushed.")
        if controller.R1:
            print("R1 has been pushed.")
        if controller.R2:
            print("R2 has been pushed.")
        if controller.LJB:
            print("LJB has been pushed.")
        if controller.RJB:
            print("RJB has been pushed.")
        if controller.options:
            print("Options has been pushed.")
        if controller.share:
            print("Share has been pushed.")
        if controller.PS:
            print("PS has been pushed.")
        if controller.pad:
            print("Pad has been pushed.")

        print("Left joystick: {}".format(controller.LJ[0]))
