#!/usr/bin/env python
"""Define the Bridge between the mouse-keyboard interface and the world.

Dependencies:
- `pyrobolearn.tools.interfaces.MouseKeyboardInterface`
- `pyrobolearn.tools.bridges.Bridge`
"""

import numpy as np

from pyrobolearn.tools.interfaces import MouseKeyboardInterface
from pyrobolearn.tools.bridges import Bridge
from pyrobolearn.robots.quadcopter import Quadcopter
from pyrobolearn.worlds.world_camera import WorldCamera

from pyrobolearn.utils.transformation import get_rpy_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BridgeMouseKeyboardQuadcopter(Bridge):
    r"""Bridge between MouseKeyboard and Quadcopter

    Bridge between the mouse-keyboard and a quadcopter robot.

    Mouse:
        * predefined in simulator:
            * `scroll wheel`: zoom
            * `ctrl`/`alt` + `scroll button`: move the camera using the mouse
            * `ctrl`/`alt` + `left-click`: rotate the camera using the mouse
            * `left-click` and drag: transport the object

    Keyboard:
        * `top arrow`: move forward
        * `bottom arrow`: move backward
        * `left arrow`: move sideways to the left
        * `right arrow`: move sideways to the right
        * `ctrl + top arrow`: ascend
        * `ctrl + bottom arrow`: descend
        * `ctrl + left arrow`: turn to the right
        * `ctrl + right arrow`: turn to the left
        * `space`: switch between first-person and third-person view.
        * predefined in simulator:
            * `w`: show the wireframe (collision shapes)
            * `s`: show the reference system
            * `v`: show bounding boxes
            * `g`: show/hide parts of the GUI the side columns
            * `esc`: quit the simulator
    """

    def __init__(self, quadcopter, interface=None, camera=None, first_person_view=False, speed=10,
                 priority=None, verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and a quadcopter.

        Args:
            quadcopter (Quadcopter): quadcopter robot instance.
            interface (None, MouseKeyboardInterface): mouse keyboard interface. If None, it will create one.
            camera (WorldCamera): world camera instance. This will allow the user to switch between first-person and
                third-person view. If None, it will create an instance of it.
            first_person_view (bool): if True, it will set the world camera to the first person view. If False, it
                will be the third-person view.
            speed (float): speed of the propeller
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # set quadcopter
        self.quadcopter = quadcopter
        if speed == 0:
            speed = 1
        self.speed = speed if speed > 0 else -speed

        # if interface not defined, create one.
        if not isinstance(interface, MouseKeyboardInterface):
            interface = MouseKeyboardInterface(self.simulator)

        # call superclass
        super(BridgeMouseKeyboardQuadcopter, self).__init__(interface, priority)

        # camera
        self.camera = camera
        self.verbose = verbose
        self.fpv = first_person_view
        self.camera_pitch = self.camera.pitch

    ##############
    # Properties #
    ##############

    @property
    def quadcopter(self):
        """Return the quadcopter instance."""
        return self._quadcopter

    @quadcopter.setter
    def quadcopter(self, quadcopter):
        """Set the quadcopter instance."""
        if not isinstance(quadcopter, Quadcopter):
            raise TypeError("Expecting the given 'quadcopter' to be an instance of `Quadcopter`, instead got: "
                            "{}".format(type(quadcopter)))
        self._quadcopter = quadcopter

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self._quadcopter.simulator

    @property
    def camera(self):
        """Return the world camera instance."""
        return self._camera

    @camera.setter
    def camera(self, camera):
        """Set the world camera instance."""
        if camera is None:
            camera = WorldCamera(self.simulator)
        elif not isinstance(camera, WorldCamera):
            raise TypeError("Expecting the given 'camera' to be an instance of `WorldCamera`, instead got: "
                            "{}".format(type(camera)))
        self._camera = camera

    ###########
    # Methods #
    ###########

    def step(self, update_interface=False):
        """Perform a step: map the mouse-keyboard interface to the world"""
        # update interface
        if update_interface:
            self.interface()

        # check keyboard events
        self.check_key_events()

        # set camera view
        pitch, yaw = get_rpy_from_quaternion(self.quadcopter.orientation)[1:]
        if self.fpv:  # first-person view
            target_pos = self.quadcopter.position + 2 * np.array([np.cos(yaw) * np.cos(pitch),
                                                                  np.sin(yaw) * np.cos(pitch),
                                                                  np.sin(pitch)])
            self.camera.reset(distance=2, pitch=-pitch, yaw=yaw - np.pi / 2, target_position=target_pos)
        else:  # third-person view
            self.camera.follow(body_id=self.quadcopter.id, distance=2, yaw=yaw - np.pi / 2, pitch=self.camera_pitch)

    def change_camera_view(self):
        """Change camera view between first-person view and third-person view."""
        self.fpv = not self.fpv

    def check_key_events(self):
        key, pressed, down = self.interface.key, self.interface.key_pressed, self.interface.key_down

        # change camera view
        if key.space in pressed:
            self.change_camera_view()

        # move the quadcopter
        if key.ctrl in down:
            if key.top_arrow in down:
                self.quadcopter.ascend(speed=10 * self.speed)
            elif key.bottom_arrow in down:
                self.quadcopter.descend(speed=10 * self.speed)
            elif key.left_arrow in down:
                self.quadcopter.turn_left(speed=self.speed)
            elif key.right_arrow in down:
                self.quadcopter.turn_right(speed=self.speed)
            else:
                self.quadcopter.hover()
        else:
            if key.top_arrow in down:
                self.quadcopter.move_forward(speed=self.speed)
            elif key.bottom_arrow in down:
                self.quadcopter.move_backward(speed=self.speed)
            elif key.left_arrow in down:
                self.quadcopter.move_left(speed=self.speed)
            elif key.right_arrow in down:
                self.quadcopter.move_right(speed=self.speed)
            else:
                self.quadcopter.hover()


# Tests
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create World
    world = prl.worlds.BasicWorld(sim)

    # load robot
    # robot = world.load_robot('quadcopter')
    robot = Quadcopter(sim, position=[0, 0, 1.])

    # create bridge/interface
    bridge = BridgeMouseKeyboardQuadcopter(robot, verbose=True)

    for _ in count():
        bridge.step(update_interface=True)
        world.step(sleep_dt=sim.dt)
