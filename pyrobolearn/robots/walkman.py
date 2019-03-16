#!/usr/bin/env python
"""Provide the WALK-MAN robotic platform.
"""

import os
from legged_robot import BipedRobot
from manipulator import BiManipulatorRobot
from sensors import CameraSensor


class Walkman(BipedRobot, BiManipulatorRobot):
    r"""Walk-man robot

    The Walk-man robot is a humanoid robot developed at the Italian Institute of Technology (IIT) with ... degrees
    of freedom, two stereo-cameras, 1 hokuyo depth laser sensor, 4 force/torque sensors (one in each wrist and one in
    each ankle).

    References:
        [1] "WALK-MAN: A High-Performance Humanoid Platform for Realistic Environments", Tsagarakis et al., 2017
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 1.14),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/walkman/walkman.urdf',
                 lower_body=False):  # 'walkman_lower_body.urdf'
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 1.14)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (1.14,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False
        if lower_body:
            urdf_path = os.path.dirname(__file__) + '/urdfs/walkman/walkman_lower_body.urdf'

        super(Walkman, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'walkman'

        # Camera sensors: Two 2D Camera sensor (stereo-camera)
        # links: "left_camera_frame", "right_camera_frame"
        # fovx = 1.3962634rad = 80 degrees, Gaussian noise = N(0, 0.007)
        # "left_camera_frame",
        self.left_camera = CameraSensor(self.sim, self.id, 11, width=800, height=800, fovy=80, near=0.02, far=300,
                                        refresh_rate=30)  # 11
        self.right_camera = CameraSensor(self.sim, self.id, 13, width=800, height=800, fovy=80, near=0.02, far=300,
                                        refresh_rate=30)  # 13

        # Laser (depth) sensor: Hokuyo sensor
        # link: "head_hokuyo_frame"
        # freq=40, samples = 720, angle = [-1.570796, 1.570796] rad, range = [0.10, 30.0] m
        # Gaussian noise: N(0.0, 0.01)
        self.depth_camera = CameraSensor(self.sim, self.id, 10, width=200, height=200, fovy=80, near=0.1, far=30.0,
                                         refresh_rate=40)

        self.cameras = [self.left_camera, self.right_camera, self.depth_camera]

        # F/T sensors

        # IMU sensors

        # End-effectors (arms and legs)
        self.waist = [self.getLinkIds(link) for link in ['DWL', 'DWS', 'DWYTorso'] if link in self.link_names]
        self.neck = [self.getLinkIds(link) for link in ['NeckYaw', 'NeckPitch'] if link in self.link_names]

        self.legs = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot'],
                                   ['RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot']]]

        self.feet = [self.getLinkIds(link) for link in ['LFoot', 'RFoot'] if link in self.link_names]

        self.arms = [[self.getLinkIds(link) for link in links if link in self.link_names]
                     for links in [['LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3'],
                                   ['RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2', 'RWrMot3']]]

        self.hands = [self.getLinkIds(link) for link in ['LSoftHand', 'RSoftHand'] if link in self.link_names]


def WalkmanLowerBody(simulator, init_pos=(0, 0, 1.5), init_orient=(0, 0, 0, 1), useFixedBase=False, scaling=1.,
                       urdf_path=os.path.dirname(__file__) + '/urdfs/walkman/walkman_lower_body.urdf'):
    """Load Walkman Lower Body"""
    return Walkman(simulator=simulator, init_pos=init_pos, init_orient=init_orient, useFixedBase=useFixedBase,
                    scaling=scaling, urdf_path=urdf_path)


# Test
if __name__ == "__main__":
    # Imports
    from itertools import count
    from pyrobolearn.simulators import BulletSim
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # Create world
    world = BasicWorld(sim)
    world.loadSphere([2., 0, 2.], mass=0., color=(1, 0, 0, 1))
    world.loadSphere([2., 1., 2.], mass=0., color=(0, 0, 1, 1))

    # load robot
    robot = Walkman(sim, useFixedBase=False, lower_body=False)

    # print information about the robot
    robot.printRobotInfo()

    # # Position control using sliders
    robot.addJointSlider(robot.left_leg)

    # run simulator
    for i in count():
        robot.updateJointSlider()
        if i % 60 == 0:
            img = robot.left_camera.getRGBImage()

        world.step(sleep_dt=1./240)
