#!/usr/bin/env python
"""Define the Bridge between the game controller interface and the world.

Dependencies:
- `pyrobolearn.tools.interfaces.controllers.controller`
- `pyrobolearn.tools.bridges.Bridge`
"""

import numpy as np

from pyrobolearn.tools.interfaces.controllers.controller import GameControllerInterface
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


class BridgeControllerQuadcopter(Bridge):
    r"""Bridge between game Controller and Quadcopter

    Bridge between the game controller and a quadcopter robot.

    Here is the mapping between the controller and the quadcopter:
    - left joystick: use to move the quadcopter
    - right joystick: use to ascend/descend and turn
    - south button (X on PlayStation and A on Xbox): change between the first-person and third-person view.
    - east button (circle on PlayStation and B on Xbox): increase the speed
    - west button (square on PlayStation and X on Xbox): decrease the speed
    """

    def __init__(self, quadcopter, interface, camera=None, first_person_view=False, speed=10,
                 priority=None, verbose=False):
        """
        Initialize the Bridge between a game controller interface and a quadcopter.

        Args:
            quadcopter (Quadcopter): quadcopter robot instance.
            interface (None, GameControllerInterface): game controller interface.
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

        # check interface
        if not isinstance(interface, GameControllerInterface):
            raise TypeError("Expecting the given 'interface' to be an instance of `GameControllerInterface`, instead "
                            "got: {}".format(type(interface)))

        # call superclass
        super(BridgeControllerQuadcopter, self).__init__(interface, priority)

        # camera
        self.camera = camera
        self.verbose = verbose
        self.fpv = first_person_view
        self.camera_pitch = self.camera.pitch

        # joystick threshold (to remove noise)
        self.threshold = 0.05

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

        # check interface events
        self.check_events()

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

    def check_events(self):
        # move the quadcopter
        left_joystick = self.interface.LJ   # (x,y)
        right_joystick = self.interface.RJ  # (x,y)
        south_button = self.interface.BTN_SOUTH
        east_button = self.interface.BTN_EAST
        west_button = self.interface.BTN_WEST
        left_norm, right_norm = np.linalg.norm(left_joystick), np.linalg.norm(right_joystick)

        # change camera view
        if south_button:
            self.change_camera_view()

        # change speed
        if east_button:
            self.speed += 1
        if west_button:
            self.speed -= 1

        # move the quadcopter
        if left_norm > self.threshold:  # left joystick
            if right_norm > self.threshold:  # with right joystick
                velocity = self.speed * np.array([left_joystick[1], left_joystick[0], right_joystick[1]])
            else:
                velocity = self.speed * np.array([left_joystick[1], left_joystick[0], 0.])
            self.quadcopter.move(velocity=velocity)
        elif right_norm > self.threshold:  # right joystick
            if np.abs(right_joystick[1]) > np.abs(right_joystick[0]):
                self.quadcopter.move(velocity=self.speed * np.array([0, 0, right_joystick[1]]))
            else:
                self.quadcopter.turn(speed=-1 * right_joystick[0])
        else:
            self.quadcopter.hover()


# Tests
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl
    from pyrobolearn.tools.interfaces.controllers.playstation import PSControllerInterface

    # create simulator
    sim = prl.simulators.Bullet()

    # create World
    world = prl.worlds.BasicWorld(sim)

    # load robot
    # robot = world.load_robot('quadcopter')
    robot = Quadcopter(sim, position=[0, 0, 2.])

    # create bridge/interface
    controller = PSControllerInterface(use_thread=True, sleep_dt=0.01)
    bridge = BridgeControllerQuadcopter(robot, interface=controller, verbose=True)

    for _ in count():
        bridge.step(update_interface=False)  # when using thread for the interface, it updates itself automatically
        world.step(sleep_dt=sim.dt)
