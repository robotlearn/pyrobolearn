#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridges between controller interface and wheeled robots
"""

from abc import ABCMeta
import numpy as np

from pyrobolearn.tools.interfaces.controllers.controller import GameControllerInterface
from pyrobolearn.tools.bridges.bridge import Bridge
from pyrobolearn.robots.wheeled_robot import WheeledRobot, DifferentialWheeledRobot, AckermannWheeledRobot
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


class BridgeControllerWheeledRobot(Bridge):
    r"""Bridge between GameController and a wheeled robot

    Bridge between the game controller and a wheeled robot.

    Here is the mapping between the controller and the robot:
    - left joystick: velocity of the wheeled robot
    - south button (X on PlayStation and A on Xbox): change between the first-person and third-person view  # TODO
    - east button (circle on PlayStation and B on Xbox): increase the speed  # TODO
    - west button (square on PlayStation and X on Xbox): decrease the speed  # TODO
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, interface=None, camera=None, first_person_view=False, speed=10,
                 priority=None, verbose=False):
        """
        Initialize the Bridge between a game controller interface and a wheeled robot instance.

        Args:
            robot (WheeledRobot): wheeled robot instance.
            interface (GameControllerInterface): game controller interface.
            camera (WorldCamera): world camera instance. This will allow the user to switch between first-person and
                third-person view. If None, it will create an instance of it.
            first_person_view (bool): if True, it will set the world camera to the first person view. If False, it
                will be the third-person view.
            speed (float): speed of the propeller
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        # set robot
        self.robot = robot
        if speed == 0:
            speed = 1
        self.speed = speed if speed > 0 else -speed

        # if interface not defined, create one.
        if not isinstance(interface, GameControllerInterface):
            raise TypeError("Expecting the given 'interface' to be an instance of `GameControllerInterface`, instead "
                            "got: {}".format(type(interface)))

        # call superclass
        super(BridgeControllerWheeledRobot, self).__init__(interface, priority=priority, verbose=verbose)

        # camera
        self.camera = camera
        self.fpv = first_person_view
        self.camera_pitch = self.camera.pitch

        # joystick threshold (to remove noise)
        self.threshold = 0.1

    ##############
    # Properties #
    ##############

    @property
    def robot(self):
        """Return the wheeled robot instance."""
        return self._robot

    @robot.setter
    def robot(self, robot):
        """Set the wheeled robot instance."""
        if not isinstance(robot, WheeledRobot):
            raise TypeError("Expecting the given 'robot' to be an instance of `WheeledRobot`, instead got: "
                            "{}".format(type(robot)))
        self._robot = robot

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self._robot.simulator

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
        pitch, yaw = get_rpy_from_quaternion(self.robot.orientation)[1:]
        if self.fpv:  # first-person view
            target_pos = self.robot.position + 2 * np.array([np.cos(yaw) * np.cos(pitch),
                                                             np.sin(yaw) * np.cos(pitch),
                                                             np.sin(pitch)])
            self.camera.reset(distance=2, pitch=-pitch, yaw=yaw - np.pi / 2, target_position=target_pos)
        else:  # third-person view
            self.camera.follow(body_id=self.robot.id, distance=2, yaw=yaw - np.pi / 2, pitch=self.camera_pitch)

    def change_camera_view(self):
        """Change camera view between first-person view and third-person view."""
        self.fpv = not self.fpv

    def check_key_events(self):
        left_joystick = self.interface.LJ[::-1]  # (y,x)
        south_button = self.interface.BTN_SOUTH
        east_button = self.interface.BTN_EAST
        west_button = self.interface.BTN_WEST

        # change camera view
        # if south_button:
        #     self.change_camera_view()

        # change speed
        if east_button:
            self.speed += 1
        if west_button:
            self.speed -= 1

        # move robot
        if np.linalg.norm(left_joystick) > self.threshold:
            # print(left_joystick)
            self.robot.move(velocity=self.speed * left_joystick)
        else:
            self.robot.move(velocity=[0., 0.])


class BridgeControllerDifferentialWheeledRobot(BridgeControllerWheeledRobot):
    r"""Bridge between the mouse-keyboard and a differential wheeled robot.

    Here is the mapping between the controller and the robot:
    - left joystick: velocity of the wheeled robot
    - south button (X on PlayStation and A on Xbox): change between the first-person and third-person view.
    - east button (circle on PlayStation and B on Xbox): increase the speed
    - west button (square on PlayStation and X on Xbox): decrease the speed
    """

    def __init__(self, robot, interface=None, camera=None, first_person_view=False, speed=10, priority=None,
                 verbose=False):
        """
        Initialize the Bridge between a game controller interface and a differential wheeled robot instance.

        Args:
            robot (AckermannWheeledRobot): wheeled robot instance.
            interface (GameControllerInterface): game controller interface.
            camera (WorldCamera): world camera instance. This will allow the user to switch between first-person and
                third-person view. If None, it will create an instance of it.
            first_person_view (bool): if True, it will set the world camera to the first person view. If False, it
                will be the third-person view.
            speed (float): speed of the propeller
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        if not isinstance(robot, DifferentialWheeledRobot):
            raise TypeError("Expecting the given 'robot' to be an instance of `DifferentialWheeledRobot`, instead "
                            "got: {}".format(type(robot)))
        super(BridgeControllerDifferentialWheeledRobot, self).__init__(robot, interface=interface, camera=camera,
                                                                       first_person_view=first_person_view,
                                                                       speed=speed, priority=priority,
                                                                       verbose=verbose)

    def check_key_events(self):
        super(BridgeControllerDifferentialWheeledRobot, self).check_key_events()
        directional_pad = self.interface.Dpad[::-1]  # (y,x)

        # move robot
        if directional_pad[0] != 0:
            self.robot.turn(directional_pad[0])
        if directional_pad[1] != 0:
            self.robot.drive_forward(directional_pad[1] * self.speed)


class BridgeControllerAckermannWheeledRobot(BridgeControllerWheeledRobot):
    r"""Bridge between the mouse-keyboard and a Ackermann wheeled robot.

    Here is the mapping between the controller and the robot:
    - left joystick: velocity of the wheeled robot
    - south button (X on PlayStation and A on Xbox): change between the first-person and third-person view.
    - east button (circle on PlayStation and B on Xbox): increase the speed
    - west button (square on PlayStation and X on Xbox): decrease the speed
    """

    def __init__(self, robot, interface, camera=None, first_person_view=False, speed=10, priority=None,
                 verbose=False):
        """
        Initialize the Bridge between a game controller interface and a wheeled robot instance.

        Args:
            robot (DifferentialWheeledRobot): wheeled robot instance.
            interface (GameControllerInterface): game controller interface.
            camera (WorldCamera): world camera instance. This will allow the user to switch between first-person and
                third-person view. If None, it will create an instance of it.
            first_person_view (bool): if True, it will set the world camera to the first person view. If False, it
                will be the third-person view.
            speed (float): speed of the propeller
            priority (int): priority of the bridge.
            verbose (bool): If True, print information on the standard output.
        """
        if not isinstance(robot, AckermannWheeledRobot):
            raise TypeError("Expecting the given 'robot' to be an instance of `AckermannWheeledRobot`, instead got: "
                            "{}".format(type(robot)))
        super(BridgeControllerAckermannWheeledRobot, self).__init__(robot, interface=interface, camera=camera,
                                                                    first_person_view=first_person_view,
                                                                    speed=speed, priority=priority, verbose=verbose)


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
    # robot = world.load_robot('epuck')
    robot = prl.robots.Epuck(sim)

    # create bridge/interface
    interface = PSControllerInterface(use_thread=True, sleep_dt=0.01)
    bridge = BridgeControllerWheeledRobot(robot, interface=interface, verbose=True)

    # run simulator
    for _ in count():
        bridge.step(update_interface=False)  # when using thread for the interface, it updates itself automatically
        world.step(sleep_dt=sim.dt)
