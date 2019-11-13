#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the mouse-keyboard interface.

Dependencies:
- `pyrobolearn.tools.interfaces`
- `pyrobolearn.simulators` (only for the mouse keyboard interface; we get the events from the simulator)
"""

import numpy as np

from pyrobolearn.utils.bullet_utils import Key, Mouse
from pyrobolearn.tools.interfaces import InputInterface
from pyrobolearn.simulators import Simulator
# from pyrobolearn.worlds import World

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MouseKeyboardInterface(InputInterface):
    r"""Mouse Keyboard Interface for the simulator

    Provide the mouse keyboard interface which can for instance allows the user to interact with the world
    (and thus the robots) using the mouse and keyboard.

    For IK, you can perform it:
    * from one link to another link: select the first link by a `left-click` then the second link that you wish to move
                                     with a `right-click`
    * using the full-body: do not select any objects, and just `right-click` on the corresponding link
    You can also use the sliders to move the corresponding link. When using the mouse, it will move the selected link
    in the plane which is parallel to the camera (i.e. perpendicular to the axis that goes from the camera lens to
    the link), and contains the CoM of the selected link.

    Mouse:
    * predefined in pybullet
        * `scroll wheel`: zoom
        * `ctrl`/`alt` + `scroll button`: move the camera using the mouse
        * `ctrl`/`alt` + `left-click`: rotate the camera using the mouse
        * `left-click` and drag: transport the object

    Keyboard:
    * predefined in pybullet:
        * `w`: show the wireframe (collision shapes)
        * `s`: show the reference system
        * `v`: show bounding boxes
        * `g`: show/hide parts of the GUI the side columns (check `sim.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)`)
        * `esc`: quit the simulator

    For programmers:
        In Pybullet, `sim.getMouseEvents()` return a list of mouse events in the following format:
        [(eventType, x, y, buttonIndex, buttonState)]
        where
        * `eventType` is 1 if the mouse is moving, 2 if a button has been pressed or released
        * `(x,y)` are the mouse position on the screen (expressed in pixel, and returned as float (they should be int))
        * `buttonIndex` is -1 if nothing, 0 if left button, 1 if scroll wheel (pressed), 2 if right button
        * `buttonState` is 0 if nothing, 3 if the button has been pressed, 4 is the button has been released,
            1 if the key is down (never observed), 2 if the key has been triggered (never observed).

        In Pybullet, `sim.getKeyboardEvents()` return a dictionary of key events in the following format:
            {keyID: keyState}
        where
        * `keyID` is an integer (ascii code) representing the key. Some special keys like shift, arrows, and others are
        are defined in pybullet such as `B3G_SHIFT`, `B3G_LEFT_ARROW`, `B3G_UP_ARROW`,...
        * `keyState` is an integer. 3 if the button has been pressed, 1 if the key is down, 2 if the key has been
        triggered, 4 if it has been released.
    """

    def __init__(self, simulator=None, verbose=False, *args, **kwargs):
        """
        Initialize the Mouse-Keyboard Interface. This interface is a little bit special in the sense that we use
        the simulator to provide the mouse and keyboard events instead of using an external library.

        Args:
            simulator (Simulator, None): simulator instance from which we capture mouse and keyboard events. If None,
                it will look for the first instantiated simulator.
            verbose (bool): If True, print information on the standard output.
        """
        super(MouseKeyboardInterface, self).__init__(use_thread=False, sleep_dt=0, verbose=verbose)

        # set simulator
        self.simulator = simulator

        # define variables for mouse events
        # Note: the difference between pressed and moment, is that a key is pressed during a short instant,
        # while the key can be down for a long period of time.
        self.mouse_moving = False
        self.mouse_pressed = False
        self.left_click_pressed = False
        self.right_click_pressed = False
        self.left_click_down = False
        self.right_click_down = False
        self.mouse_x, self.mouse_y = 0, 0

        # define variables for key events
        self.key_pressed = set([])
        self.key_down = set([])
        self.key = Key

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self._simulator

    @simulator.setter
    def simulator(self, simulator):
        """Set the simulator instance."""
        if simulator is None:
            if len(Simulator.instances) == 0:
                raise RuntimeError("No simulator was given to the `MouseKeyboardInterface`, we thus tried to look "
                                   "for an instantiated simulator but none was found...")
            simulator = Simulator.instances[0]  # by default, we take the first instantiated simulator
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))
        self._simulator = simulator

    @property
    def mouse_pressed(self):
        """Return True if one of the mouse buttons has been pressed."""
        return self.left_click_pressed or self.right_click_pressed

    @mouse_pressed.setter
    def mouse_pressed(self, pressed):
        """Set if the mouse has been pressed."""
        self.left_click_pressed = pressed
        self.right_click_pressed = pressed

    @property
    def mouse_down(self):
        """Check if one of the mouse buttons is down."""
        return self.left_click_down or self.right_click_down

    ###########
    # Methods #
    ###########

    def check_key_events(self):
        """Check the key events."""
        # get key events
        events = self.simulator.get_keyboard_events()

        # create new list of key pressed
        self.key_pressed, self.key_down = set([]), set([])
        if Key.shift in events:
            self.key_pressed.add(Key.shift)
            self.key_down.add(Key.shift)
        if Key.alt in events:
            self.key_pressed.add(Key.alt)
            self.key_down.add(Key.alt)
        if Key.ctrl in events:
            self.key_pressed.add(Key.ctrl)
            self.key_down.add(Key.ctrl)

        # go through each keyboard event
        for key, state in events.items():
            if state == Key.pressed:  # if the key is pressed
                self.key_pressed.add(key)
                self.key_down.add(key)  # a key pressed is also a key down
            elif state == Key.down:  # if the key is down
                self.key_down.add(key)

    def check_mouse_events(self):
        """Check the mouse events."""
        # get mouse events
        events = self.simulator.get_mouse_events()

        # reset mouse events
        self.mouse_moving, self.mouse_pressed = False, False

        # go through each mouse event
        for event in events:
            eventType, x, y, idx, state = event
            self.mouse_x, self.mouse_y = x, y

            # check if mouse is moving
            if eventType == Mouse.moving:
                self.mouse_moving = True

            # check if button has been pressed or released
            elif eventType == Mouse.button:
                # check left click
                if idx == Mouse.left_click:
                    if state == Mouse.pressed:
                        self.left_click_pressed = True
                        self.left_click_down = True
                    elif state == Mouse.released:
                        self.left_click_pressed = False
                        self.left_click_down = False

                # check right click
                elif idx == Mouse.right_click:
                    if state == Mouse.pressed:
                        self.right_click_pressed = True
                        self.right_click_down = True
                    elif state == Mouse.released:
                        self.right_click_pressed = False
                        self.right_click_down = False

    def step(self):
        """Perform a step with the interface."""
        # check key events
        self.check_key_events()

        # check mouse events
        self.check_mouse_events()

        if self.verbose:
            print("\nKey pressed: {}".format(self.key_pressed))
            print("Mouse moving? {}".format(self.mouse_moving))
            print("Mouse pressed? {}".format(self.mouse_pressed))
            print("Mouse down? {}".format(self.mouse_down))


# Tests
if __name__ == '__main__':
    import time
    from itertools import count
    from pyrobolearn.simulators import Bullet

    # create simulator
    sim = Bullet()

    # create interface
    interface = MouseKeyboardInterface(sim, verbose=True)

    for _ in count():
        interface.step()
        time.sleep(1. / 2)
