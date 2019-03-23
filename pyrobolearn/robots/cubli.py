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
                 position=(0, 0, 0.5),
                 orientation=(0, 0, 0, 1),
                 fixed_base=False,
                 scaling=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/cubli/cubli.urdf'):
        # check parameters
        if position is None:
            position = (0., 0., 0.5)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (0.5,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(Cubli, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
        self.name = 'cubli'

        # disable each motor joint
        self.disable_motor()


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
    robot.print_info()
    H = robot.get_mass_matrix(q_idx=slice(6, 6 + len(robot.joints)))  # floating base, thus keep only the last q
    print("Inertia matrix: H(q) = {}\n".format(H))

    # PD control
    Kp = 600.
    Kd = 2 * np.sqrt(Kp)
    desired_roll = np.pi / 4.

    for i in count():
        # get state
        quaternion = robot.get_base_orientation(False)
        w = robot.get_base_angular_velocity()
        euler = pybullet.getEulerFromQuaternion(quaternion.tolist())

        # PD control
        torques = [-Kp * (desired_roll - euler[0]) + Kd * w[0], 0., 0.]
        robot.set_joint_torques(torques)

        # step in simulation
        world.step(sleep_dt=1./240)
