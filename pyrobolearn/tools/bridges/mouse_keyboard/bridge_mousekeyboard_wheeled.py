# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the Bridge between the mouse-keyboard interface and the world.

Dependencies:
- `pyrobolearn.tools.interfaces.MouseKeyboardInterface`
- `pyrobolearn.tools.bridges.Bridge`
"""

from abc import ABCMeta
import numpy as np

from pyrobolearn.tools.interfaces import MouseKeyboardInterface
from pyrobolearn.tools.bridges import Bridge
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


class BridgeMouseKeyboardWheeledRobot(Bridge):
    r"""Bridge between MouseKeyboard and a wheeled robot

    Bridge between the mouse-keyboard and a wheeled robot.

    Mouse:
        * predefined in simulator:
            * `scroll wheel`: zoom
            * `ctrl`/`alt` + `scroll button`: move the camera using the mouse
            * `ctrl`/`alt` + `left-click`: rotate the camera using the mouse
            * `left-click` and drag: transport the object

    Keyboard:
        * `top arrow`: move forward
        * `bottom arrow`: move backward
        * `left arrow`: turn/steer to the left
        * `right arrow`: turn/steer to the right
        * `space`: switch between first-person and third-person view.  # TODO
        * `shift`: increase speed
        * `ctrl`: decrease speed
        * predefined in simulator:
            * `w`: show the wireframe (collision shapes)
            * `s`: show the reference system
            * `v`: show bounding boxes
            * `g`: show/hide parts of the GUI the side columns
            * `esc`: quit the simulator
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, interface=None, camera=None, first_person_view=False, speed=10,
                 priority=None, verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and a wheeled robot instance.

        Args:
            robot (WheeledRobot): wheeled robot instance.
            interface (None, MouseKeyboardInterface): mouse keyboard interface. If None, it will create one.
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
        if not isinstance(interface, MouseKeyboardInterface):
            interface = MouseKeyboardInterface(self.simulator)

        # call superclass
        super(BridgeMouseKeyboardWheeledRobot, self).__init__(interface, priority)

        # camera
        self.camera = camera
        self.verbose = verbose
        self.fpv = first_person_view
        self.camera_pitch = self.camera.pitch

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
        key, pressed, down = self.interface.key, self.interface.key_pressed, self.interface.key_down

        # change camera view
        # if key.space in pressed:
        #     self.change_camera_view()

        if key.shift in pressed:
            self.speed += 1
        if key.ctrl in pressed:
            self.speed -= 1


class BridgeMouseKeyboardDifferentialWheeledRobot(BridgeMouseKeyboardWheeledRobot):
    r"""Bridge between the mouse-keyboard and a differential wheeled robot.

    Keyboard:
        * `top arrow`: move forward
        * `bottom arrow`: move backward
        * `left arrow`: turn to the left
        * `right arrow`: turn to the right
        * `space`: switch between first-person and third-person view.
        * predefined in simulator:
            * `w`: show the wireframe (collision shapes)
            * `s`: show the reference system
            * `v`: show bounding boxes
            * `g`: show/hide parts of the GUI the side columns
            * `esc`: quit the simulator
    """

    def __init__(self, robot, interface=None, camera=None, first_person_view=False, speed=10, priority=None,
                 verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and a differential wheeled robot instance.

        Args:
            robot (DifferentialWheeledRobot): wheeled robot instance.
            interface (None, MouseKeyboardInterface): mouse keyboard interface. If None, it will create one.
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
        super(BridgeMouseKeyboardDifferentialWheeledRobot, self).__init__(robot, interface=interface, camera=camera,
                                                                          first_person_view=first_person_view,
                                                                          speed=speed, priority=priority,
                                                                          verbose=verbose)

    def check_key_events(self):
        super(BridgeMouseKeyboardDifferentialWheeledRobot, self).check_key_events()
        key, pressed, down = self.interface.key, self.interface.key_pressed, self.interface.key_down

        # move the robot
        if key.top_arrow in down:
            self.robot.drive(speed=self.speed)
        elif key.bottom_arrow in down:
            self.robot.drive(speed=-self.speed)
        elif key.left_arrow in down:
            self.robot.turn(speed=self.speed/10.)
        elif key.right_arrow in down:
            self.robot.turn(speed=-self.speed/10.)
        else:
            self.robot.drive(speed=0)


class BridgeMouseKeyboardAckermannWheeledRobot(BridgeMouseKeyboardWheeledRobot):
    r"""Bridge between the mouse-keyboard and a Ackermann wheeled robot.

    Keyboard:
        * `top arrow`: move forward
        * `bottom arrow`: move backward
        * `left arrow`: steer to the left
        * `right arrow`: steer to the right
        * `space`: switch between first-person and third-person view.
        * predefined in simulator:
            * `w`: show the wireframe (collision shapes)
            * `s`: show the reference system
            * `v`: show bounding boxes
            * `g`: show/hide parts of the GUI the side columns
            * `esc`: quit the simulator
    """

    def __init__(self, robot, interface=None, camera=None, first_person_view=False, speed=10, priority=None,
                 verbose=False):
        """
        Initialize the Bridge between a Mouse-Keyboard interface and a wheeled robot instance.

        Args:
            robot (DifferentialWheeledRobot): wheeled robot instance.
            interface (None, MouseKeyboardInterface): mouse keyboard interface. If None, it will create one.
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
        super(BridgeMouseKeyboardAckermannWheeledRobot, self).__init__(robot, interface=interface, camera=camera,
                                                                       first_person_view=first_person_view,
                                                                       speed=speed, priority=priority, verbose=verbose)

    def check_key_events(self):
        key, pressed, down = self.interface.key, self.interface.key_pressed, self.interface.key_down

        # move the robot
        if key.top_arrow in down:
            self.robot.drive(speed=self.speed)
        elif key.bottom_arrow in down:
            self.robot.drive(speed=self.speed)
        elif key.left_arrow in down:
            self.robot.steer(speed=self.speed)
        elif key.right_arrow in down:
            self.robot.steer(speed=-self.speed)
        else:
            self.robot.drive(speed=0)


# Tests
if __name__ == '__main__':
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create World
    world = prl.worlds.BasicWorld(sim)

    # load robot
    # robot = world.load_robot('epuck')
    robot = prl.robots.Epuck(sim, position=[0, 0, 1.])

    # create bridge/interface
    bridge = BridgeMouseKeyboardDifferentialWheeledRobot(robot, verbose=True)

    for _ in count():
        bridge.step(update_interface=True)
        world.step(sleep_dt=sim.dt)
