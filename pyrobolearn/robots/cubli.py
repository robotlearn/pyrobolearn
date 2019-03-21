#!/usr/bin/env python
"""Provide the Cubli robotic platform.
"""

import os

from pyrobolearn.robots.robot import Robot


class Cubli(Robot):
    r"""Cubli robot

    References:
        [1] "The Cubli: a Cube that can Jump Up and Balance", Gajamohan et al., 2012
        [2] "The Cubli: a Reaction Wheel Based 3D Inverted Pendulum", Gajamohan et al., 2013
        [3] "Nonlinear Analysis and Control of a Reaction Wheel-based 3D Inverted Pendulum", Muehlebach et al., 2017
        [4] "Balancing Control of a Cubical Robot Balancing on its Corner", Chen et al., 2018

    Also, check about M-Blocks:
        [1] "M-Blocks: Momentum-driven, Magnetic Modular Robots", Romanishin et al., 2013
        [2] "3D M-Blocks: Self-reconfiguring Robots Capable of Locomtion via Pivoting in 3 Dimensions", Romanishin
            et al., 2015
    """

    def __init__(self,
                 simulator,
                 init_pos=(0, 0, 0.5),
                 init_orient=(0, 0, 0, 1),
                 useFixedBase=False,
                 scaling=1.,
                 urdf_path=os.path.dirname(__file__) + '/urdfs/cubli/cubli.urdf'):
        # check parameters
        if init_pos is None:
            init_pos = (0., 0., 0.5)
        if len(init_pos) == 2:  # assume x, y are given
            init_pos = tuple(init_pos) + (0.5,)
        if init_orient is None:
            init_orient = (0, 0, 0, 1)
        if useFixedBase is None:
            useFixedBase = False

        super(Cubli, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
        self.name = 'cubli'

        # disable each motor joint
        self.disableMotor()


# Test
if __name__ == "__main__":
    import numpy as np
    from itertools import count
    from pyrobolearn.simulators import BulletSim, pybullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)

    # create robot
    scale = 1.  # Warning: this does not scale the mass...
    position = [0., 0., np.sqrt(2) / 2. * scale + 0.001]
    orientation = [0.383, 0, 0, 0.924]
    robot = Cubli(sim, position, orientation, scaling=scale)

    # print information about the robot
    robot.printRobotInfo()
    H = robot.calculateMassMatrix(qIdx=slice(6, 6+len(robot.joints)))  # floating base, thus keep only the last q
    print("Inertia matrix: H(q) = {}\n".format(H))

    # PD control
    Kp = 600.
    Kd = 2 * np.sqrt(Kp)
    desired_roll = np.pi / 4.

    for i in count():
        # get state
        quaternion = robot.getBaseOrientation(False)
        w = robot.getBaseAngularVelocity()
        euler = pybullet.getEulerFromQuaternion(quaternion.tolist())

        # PD control
        torques = [-Kp * (desired_roll - euler[0]) + Kd * w[0], 0., 0.]
        robot.setJointTorques(torques)

        # step in simulation
        world.step(sleep_dt=1./240)
