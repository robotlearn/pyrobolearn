# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the reaching manipulation environment.

The goal is to reach a certain 3D (fixed or movable) target with the end-effector of a robot.
"""

import re
import numpy as np

import pyrobolearn as prl
from pyrobolearn.envs.manipulation.manipulation import ManipulationEnv


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ReachingManipulationEnv(ManipulationEnv):
    r"""Reaching Manipulation Env

    The goal is to reach a certain 3D (fixed or movable) target with the end-effector of a robot.
    """

    def __init__(self, world, target=(0.5, 0., 0.5), robot='kuka_iiwa', end_effector_id=None, control_mode='position',
                 *args, **kwargs):
        """
        Initialize the reaching manipulation environment.

        Args:
            world (World, Simulator): world/simulator instance.
            target (list/tuple of 3 floats, np.array[3], Body): target to reach. If a list, tuple, or array is
                provided, it will load a visual sphere at the specified target location.
            robot (str, Robot): robot instance, or robot name to load in the world.
            end_effector_id (int, None): end effector link id that has to reach the target. If None, it will check in
                the end_
            control_mode (str): joint action control mode, select between {'position', 'position change', 'velocity',
                'velocity change', 'torque', 'torque with gravity compensation'/'torque change'}. Note that 'torque
                change' and 'torque with gravity compensation' are the same (they are synonyms).
            args (list): list of arguments that are given to the `world.load_robot` method.
            kwargs (dict): dict of arguments that are given to the `world.load_robot` method.
        """
        # create basic world if not already created
        if isinstance(world, prl.simulators.Simulator):
            world = prl.worlds.BasicWorld(world)
        elif not isinstance(world, prl.worlds.World):
            raise TypeError("Expecting the world to be an instance of `World` or `Simulator`, instead got: "
                            "{}".format(type(world)))

        # load robot
        if isinstance(robot, str):
            robot = world.load_robot(robot, *args, **kwargs)
        elif isinstance(robot, prl.robots.Robot):
            # check if robot already loaded
            if robot.id not in world.bodies:
                world.load_robot(robot)
        else:
            raise TypeError("Expecting the given robot to be a string or an instance of `Robot`, instead got: "
                            "{}".format(type(robot)))
        self.robot = robot

        # load target
        if not isinstance(target, prl.robots.Body):
            if not isinstance(target, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the target to be list/tuple/array of 3 floats representing the target "
                                "position, or an instance of `Body`, but got instead: {}".format(type(target)))
            target = world.load_visual_sphere(position=target, radius=0.05, color=(1, 0, 0, 0.5), return_body=True)
        # save target body such that the user can use it (to change its position for instance)
        self.target = target

        # check end effector id
        if end_effector_id is None:
            if not hasattr(robot, 'end_effectors'):
                raise ValueError("We could not find any end effectors for the given robot... Please specify one by "
                                 "setting the 'end_effector_id' parameter.")
            end_effector_id = robot.end_effectors[0]
        if not isinstance(end_effector_id, int):
            raise TypeError("Expecting the 'end_effector_id' to be None or an integer, but got instead: "
                            "{}".format(type(end_effector_id)))

        # create state
        state = prl.states.LinkWorldPositionState(robot, link_ids=end_effector_id)

        # create action based on the specified joint action control mode
        control_mode = control_mode.lower()
        control_mode = ' '.join(re.findall(r"[a-z]*[^\-\_]", control_mode))
        if control_mode == 'position':
            action = prl.actions.JointPositionAction(robot)
        elif control_mode == 'position change':
            action = prl.actions.JointPositionChangeAction(robot)
        elif control_mode == 'velocity':
            action = prl.actions.JointVelocityAction(robot)
        elif control_mode == 'velocity change':
            action = prl.actions.JointVelocityChangeAction(robot)
        elif control_mode == 'torque':
            action = prl.actions.JointTorqueAction(robot)
        elif control_mode == 'torque with gravity compensation' or control_mode == 'torque change':
            action = prl.actions.JointTorqueGravityCompensationAction(robot)
        else:
            raise ValueError("Please select the `control_mode` to be between ['position', 'position change', "
                             "'velocity', 'velocity change', 'torque', "
                             "'torque with gravity compensation'/'torque change'], and not: "
                             "{}".format(control_mode))

        # create distance cost
        reward = prl.rewards.DistanceCost(state, target)

        # create environment using composition
        super(ReachingManipulationEnv, self).__init__(world=world, states=state, rewards=reward, actions=action)


# Test
if __name__ == "__main__":
    from itertools import count
    import pyrobolearn as prl

    # create simulator
    sim = prl.simulators.Bullet()

    # create environment
    env = ReachingManipulationEnv(sim)

    # run simulation
    for _ in count():
        env.step(sleep_dt=1./240)
