#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the space mouse (from 3D connexion) interface.

Prerequisites:
1. install spacenavd (http://spacenav.sourceforge.net/): `sudo apt-get install spacenavd`
2. install spnav (https://spnav.readthedocs.io/en/latest/): `pip install spnav`

Troubleshooting:
- if you get 'OSError: libspnav.so: cannot open shared object file: No such file or directory', you have to copy-paste
  the file '/usr/lib/libspnav.so.0' to '/usr/lib/libspnav.so'

Dependencies:
- `pyrobolearn.tools.interfaces`

References:
    - [1] SpaceMouse from 3D connexion: https://www.3dconnexion.com/products/spacemouse.html
    - [2] SpaceNav: http://spacenav.sourceforge.net/
    - [3] spnav: Space Navigator support in Python:
        - documentation: https://spnav.readthedocs.io/en/latest/
        - repository: https://bitbucket.org/seibert/spnav/src/default/
"""

import numpy as np

try:
    import spnav
except ImportError as e:
    string = "\nHint: try to install `spnav` by typing: " \
             "\n`sudo apt-get install spacenavd`" \
             "\n`pip install spnav`"
    raise ImportError(e.__str__() + string)


from pyrobolearn.tools.interfaces import InputInterface


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SpaceMouseInterface(InputInterface):
    r"""Space Mouse Interface

    Provide the space mouse interface which, for instance, allows the user to interact with the world and the robots
    using the space mouse (from 3D connexion).

    Space mouse:
    * 2 buttons
    * 3 degrees of translation and 3 degrees of rotation

    References:
        - [1] SpaceMouse from 3D connexion: https://www.3dconnexion.com/products/spacemouse.html
        - [2] SpaceNav: http://spacenav.sourceforge.net/
        - [3] spnav: Space Navigator support in Python:
            - documentation: https://spnav.readthedocs.io/en/latest/
            - repository: https://bitbucket.org/seibert/spnav/src/default/
    """

    def __init__(self, use_thread=False, sleep_dt=0., verbose=False, translation_range=(-1, 1),
                 rotation_range=(-1, 1)):
        """
        Initialize the SpaceMouse interface.

        Args:
            use_thread (bool): If True, it will run the interface in a separate thread than the main one.
                The interface will update its data automatically.
            sleep_dt (float): If :attr:`use_thread` is True, it will sleep the specified amount before acquiring or
                setting the next sample.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
            translation_range (np.array[2], np.array[2,3], tuple of 2 float, tuple of 2 np.array[3]): the lower and
                higher bounds for the (x, y, z). This is used to normalize the translation range to be between
                [-1, 1]. The frame (x,y,z) is defined with x pointing forward, y to the left, and z up.
            rotation_range (np.array[2], np.array[2,3], tuple of 2 float, tuple of 2 np.array[3]): the lower and higher
                bounds for the roll-pitch-yaw angles. This is used to normalize the rotation range to be between
                [-1, 1]. The frame (x,y,z) is defined with x pointing forward, y to the left, and z up.
        """

        # Open connection to the daemon via AF_UNIX socket.
        spnav.spnav_open()

        # check the ranges and their shapes
        translation_range = np.abs(np.asarray(translation_range, dtype=np.float))
        rotation_range = np.abs(np.asarray(rotation_range, dtype=np.float))

        if translation_range.shape != (2,) and translation_range.shape != (2, 3):
            raise ValueError("Expecting the shape of 'translation_range' to be (2,) or (2,3), instead got: "
                             "{}".format(translation_range.shape))
        if rotation_range.shape != (2,) and rotation_range.shape != (2, 3):
            raise ValueError("Expecting the shape of 'rotation_range' to be (2,) or (2,3), instead got: "
                             "{}".format(rotation_range.shape))

        if translation_range.shape == (2,):
            translation_range = np.vstack([translation_range]*3).T
        if rotation_range.shape == (2,):
            rotation_range = np.vstack([rotation_range]*3).T

        self.translation_range = translation_range
        self.rotation_range = rotation_range

        # define space mouse events
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
        self.updated = False

        super(SpaceMouseInterface, self).__init__(use_thread=use_thread, sleep_dt=sleep_dt, verbose=verbose)

    ###########
    # Methods #
    ###########

    def run(self):
        """Run the interface."""

        # # Remove pending space navigator events from the queue
        # spnav.spnav_remove_events(spnav.SPNAV_EVENT_ANY)

        # get space mouse event (without waiting)
        # event = spnav_wait_event()  # blocking wait for the next event
        # try:
        event = spnav.spnav_poll_event()
        # except KeyboardInterrupt:
        #     print '\nQuitting...'
        # finally:
        #     self.close()

        # check the event
        self.updated = True
        if isinstance(event, spnav.SpnavButtonEvent):  # button event
            if event.bnum == 0:  # left
                self.left_button_pressed = event.press
            elif event.bnum == 1:  # right
                self.right_button_pressed = event.press
        elif isinstance(event, spnav.SpnavMotionEvent):  # mouse motion event
            # translation
            x, y, z = event.translation                                 # x, y, z in spnav
            self.translation = np.array([z, -x, y], dtype=np.float)     # x (front), y (left), z (up)
            condition = self.translation < 0
            self.translation[condition] /= self.translation_range[0][condition]
            self.translation[~condition] /= self.translation_range[1][~condition]

            # rotation
            r, p, y = event.rotation                                # r, p, y in spnav
            self.rotation = np.array([y, -r, p], dtype=np.float)    # x (front), y (left), z (up)
            condition = self.rotation < 0
            self.rotation[condition] /= self.rotation_range[0][condition]
            self.rotation[~condition] /= self.rotation_range[1][~condition]
        else:
            self.updated = False

        # if we should print the events
        if self.verbose and self.updated:
            print("Left and Right buttons: {} and {}".format(self.left_button_pressed, self.right_button_pressed))
            print("Space mouse translation and orientation: {} and {}".format(self.translation, self.rotation))

    def close(self):
        """Close the interface."""
        # spnav.spnav_close()
        if self.use_thread:
            self.stop_thread = True


# Test the interface
if __name__ == '__main__':
    import time
    from itertools import count

    # create interface
    # interface = SpaceMouseInterface(verbose=True)
    interface = SpaceMouseInterface(verbose=True, translation_range=[[-410, -410, -430], [460, 400, 420]],
                                    rotation_range=[[-410., -420, -360], [410, 400, 410]])

    for i in count():
        interface.step()
        # if i > 1e6:
        #     print('closing the program')
        #     interface.close()
        #     break
        time.sleep(0.001)
