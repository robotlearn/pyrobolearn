# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the kendo environment.
"""

# TODO: finish to implement this environment

import pyrobolearn as prl
from pyrobolearn.worlds.samples.sports.kendo import KendoWorld
from pyrobolearn.envs.env import Env


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class KendoEnv(Env):
    r"""Kendo environment.

    """

    def __init__(self, simulator, robot='kuka_iiwa', verbose=False):
        """
        Initialize the kendo environment.

        Args:
            simulator (Simulator, None): simulator instance. If simulator is None, it will use the PyBullet simulator.
            robot (str): robot name.
            verbose (bool): if True, it will print information when creating the environment.
        """
        # check simulator
        if simulator is None:
            simulator = prl.simulators.Bullet()
        elif not isinstance(simulator, prl.simulators.Simulator):
            raise TypeError("Expecting the given 'simulator' to be an instance of `Simulator`, but got instead: "
                            "{}".format(type(simulator)))

        # create world
        world = KendoWorld(simulator, position=(0., 0., 1.5), num_shinai=2)

        # create manipulators
        robot1 = world.load_robot(robot)
        robot2 = world.load_robot(robot, position=(1., 0.), orientation=(0., 0., 1., 0.))
        self.robot1, self.robot2 = robot1, robot2

        # attach shinai to robot end effectors
        world.attach(body1=robot1, body2=world.shinai[0], link1=robot1.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.],
                     parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.15],
                     parent_frame_orientation=[0, -0.707, 0., .707])
        world.attach(body1=robot2, body2=world.shinai[1], link1=robot2.end_effectors[0], link2=-1,
                     joint_axis=[0., 0., 0.],
                     parent_frame_position=[0., 0., 0.02], child_frame_position=[0., 0., 0.15],
                     parent_frame_orientation=[0, -0.707, 0., .707])

        if not isinstance(robot1, prl.robots.Manipulator):
            raise TypeError("Expecting a manipulator, but got instead {}".format(type(robot1)))
        if verbose:
            self.robot1.print_info()

        # create states
        states = []
        for i, robot in enumerate([robot1, robot2]):
            q_state = prl.states.JointPositionState(robot=robot)
            dq_state = prl.states.JointVelocityState(robot=robot)
            state = q_state + dq_state
            states.append(state)
        if verbose:
            print(states)

        # create actions
        actions = []
        for i, robot in enumerate([robot1, robot2]):
            action = prl.actions.JointPositionAction(robot=robot, kp=robot.kp, kd=robot.kd)
            actions.append(action)
        if verbose:
            print(actions)

        # create reward  # TODO: use geometrical rewards
        reward = None

        # create terminal condition
        terminal_condition = None

        # create initial state generator
        initial_state_generator = None

        super(KendoEnv, self).__init__(world=world, states=states, rewards=reward, actions=actions,
                                       terminal_conditions=terminal_condition,
                                       initial_state_generators=initial_state_generator)


# Test
if __name__ == '__main__':
    from itertools import count

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = KendoEnv(sim)

    # run simulation
    for t in count():
        env.step(sleep_dt=sim.dt)
