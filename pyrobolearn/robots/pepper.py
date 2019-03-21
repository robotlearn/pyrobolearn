#!/usr/bin/env python
"""Provide the Pepper robotic platform.
"""

import os

from pyrobolearn.robots.wheeled_robot import WheeledRobot
from pyrobolearn.robots.manipulator import BiManipulatorRobot
from pyrobolearn.robots.sensors.camera import CameraSensor


class Pepper(WheeledRobot, BiManipulatorRobot):
    r"""Pepper robot.

    The Pepper robot is a robot from the Aldebaran company.

    Note that in the URDF, the continuous joints were replace by revolute joints. Be careful, that the limit values
    for these joints are probably not correct.

    For more information:
        [1] http://doc.aldebaran.com/2-0/home_juliette.html
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.9),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/pepper/pepper.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.9)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.9,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Pepper, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase)
        self.name = 'pepper'

        # 2D Camera sensor
        # From [1]: "Two identical video cameras are located in the forehead. They provide a resolution up to
        # 2560x1080 at 5 frames per second. VFOV = 44.30 deg, HFOV= 57.20 deg, focus = [30cm, infinity]

        # Note that we divide width and height by 4 (otherwise the simulator is pretty slow)
        self.cameraTop = CameraSensor(self.sim, self.id, 4, width=2560/4, height=1080/4, fovy=44.30,
                                      near=0.3, far=100, refresh_rate=60)
        self.cameraBottom = CameraSensor(self.sim, self.id, 9 ,width=2560/4, height=1080/4, fovy=44.30,
                                         near=0.3, far=100, refresh_rate=60)

        # 3D camera sensor
        # From [1]: "One 3D camera is located in the forehead. It provides image resolution up to 320x240 at
        # 20 frames per second. One ASUS Xtion 3D sensor is located behind the eyes. VFOV = 45 deg, HFOV = 58 deg,
        # focus = [80cm, 3.5m]."
        self.cameraDepth = CameraSensor(self.sim, self.id, 6, width=320, height=240, fovy=45, near=0.3, far=3.5,
                                        refresh_rate=120)

        self.cameras = [self.cameraTop, self.cameraBottom, self.cameraDepth]


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = Pepper(sim)

    # print information about the robot
    robot.printRobotInfo()

    # Position control using sliders
    # robot.addJointSlider()

    # run simulator
    for i in count():
        # robot.updateJointSlider()
        if i % 20 == 0:
            robot.cameraTop.getRGBImage()
        world.step(sleep_dt=1./240)
