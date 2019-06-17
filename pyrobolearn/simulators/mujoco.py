#!/usr/bin/env python
"""Define the MuJoCo Simulator API.

This is the main interface that communicates with the MuJoCo simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
MuJoCo.

Warnings: The MuJoCo simulator requires a license in order to use it.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    [1] MuJoCo: http://www.mujoco.org/
    [2] MuJoCo Python: https://github.com/openai/mujoco-py
    [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
"""

# TODO

from pyrobolearn.simulators.simulator import Simulator

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Mujoco(Simulator):
    r"""Mujoco Simulator interface.

    This is the main interface that communicates with the MuJoCo simulator [1]. By defining this interface, it allows
    to decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
    MuJoCo.

    Warnings: The MuJoCo simulator requires a license in order to use it.

    References:
        [1] MuJoCo: http://www.mujoco.org/
        [2] MuJoCo Python: https://github.com/openai/mujoco-py
        [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
    """

    def __init__(self, render=True):
        super(Mujoco, self).__init__(render=render)
        raise NotImplementedError
